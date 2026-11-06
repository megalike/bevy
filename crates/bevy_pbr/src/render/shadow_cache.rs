//! The render world's half of the shadow cache (`bevy_light`'s
//! `shadow_cache` module: [`ShadowCacheSettings`]).
//!
//! The cache keeps its state by shadow view ([`ShadowViewId`]): a spot
//! light's shadow map, and, reserved for later, a directional light's cascade
//! for a camera or a point light's cube face. Each light type's adapter
//! describes its cached views ([`CachedShadowView`]: projection, frustum,
//! destination); today only spot lights are cached
//! ([`ShadowViewCaches::prepare_spot_light`]).
//!
//! Each cached view that fits in the budget gets a static layer of its own,
//! a depth texture that holds the shadow of its settled casters. Each frame
//! [`ShadowViewCaches`] plans the view from what the main world found (its
//! static revision, where its movable casters are) and from what the view's
//! textures hold:
//!
//! | Plan | When | Views |
//! | --- | --- | --- |
//! | Kept | its shadow map is current and nothing moves in its frustum | none |
//! | Movers | movable casters are, or were last drawn, in its frustum | restore the tiles they touch now and were drawn into from the static layer, draw them |
//! | Rebuild | its static content is invalid | draw the static layer, restore it whole, draw the movable casters |
//! | Restore | its shadow map moved (another layer or texture) | restore the whole static layer, draw the movable casters |
//! | Fallback | no usable static layer (budget, or its restore's blit isn't compiled yet), and movable casters are or were in its frustum, or its map isn't current | one view draws every caster into the shadow map, as for an uncached light |
//! | Dormant | it lights no view | none; the map keeps what it holds (the tiles live casters were drawn into stay written), so the frame it lights a view again plans as after any frame |
//!
//! A view with a usable static layer draws its static casters and its
//! movable casters with two render views
//! ([`ShadowViewRole::Static`], [`ShadowViewRole::Movers`]); the static views
//! of all lights run before the movers views ([`RootNonCameraViewOrder`]), and
//! the restore runs at the start of the movers view: one instanced quad per
//! run of set tiles in a row of the view's [`ShadowTileMask`], or one
//! triangle over the whole layer. After the frame's passes
//! were recorded, [`acknowledge_shadow_caches`] checks that every caster
//! of a drawn static view was ready (its mesh, material and pipeline) and only
//! then counts the static content as valid, and reports to the main world
//! ([`ShadowCacheFeedback`], by view and generation).

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_camera::{primitives::Frustum, visibility::RenderLayers};
use bevy_core_pipeline::schedule::{Core3d, RootNonCameraView, RootNonCameraViewOrder};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel, system::SystemParam};
use bevy_light::{
    spot_light_clip_from_view, spot_light_world_from_view, ShadowCacheFaults, ShadowCacheFeedback,
    ShadowCacheSettings, ShadowCacheState, ShadowCacheViewStatus, ShadowTileMask, ShadowViewCache,
    ShadowViewId,
};
use bevy_math::{Mat4, URect, UVec2, UVec4};
use bevy_mesh::{Mesh3d, VertexBufferLayout, VertexFormat};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    diagnostic::RecordDiagnostics,
    erased_render_asset::ErasedRenderAssets,
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::RenderAssets,
    render_phase::ViewBinnedRenderPhases,
    render_resource::{binding_types::texture_depth_2d, *},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::{MainEntity, MainEntityHashSet},
    texture::{CachedTexture, DepthStencilViewAttachment, DepthStencilViews},
    view::{ExtractedView, NoIndirectDrawing, RenderShadowMapVisibleEntities, RetainedViewEntity},
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::ShaderCacheError;
use bevy_transform::components::GlobalTransform;
use bevy_utils::default;

use crate::{
    shadow_map_visible_entities_key, ExtractedPointLight, LightEntity, PendingShadowQueues,
    PointAndSpotLightViewEntities, PreparedMaterial, PreprocessBindGroups, PreprocessPipelines,
    RenderMaterialInstances, RenderMeshInstanceFlags, RenderMeshInstances, Shadow, ShadowView,
    SpecializedShadowMaterialPipelineCache, CORE_3D_SHADOW_MAP_FORMAT,
};

/// Adds the render world's half of the shadow cache.
pub struct ShadowCachePlugin;

impl Plugin for ShadowCachePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shadow_restore.wesl");

        let feedback = app
            .world_mut()
            .get_resource_or_init::<ShadowCacheFeedback>()
            .clone();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(feedback)
            .init_resource::<ShadowViewCaches>()
            .init_resource::<ShadowCacheSettings>()
            .add_systems(RenderStartup, init_shadow_restore_pipeline)
            .add_systems(ExtractSchedule, extract_shadow_cache_settings)
            .add_systems(
                Render,
                acknowledge_shadow_caches.in_set(RenderSystems::Cleanup),
            );
    }
}

fn extract_shadow_cache_settings(
    mut commands: Commands,
    settings: Extract<Res<ShadowCacheSettings>>,
    faults: Extract<Option<Res<ShadowCacheFaults>>>,
) {
    if settings.is_changed() {
        commands.insert_resource(settings.clone());
    }
    commands.insert_resource(faults.as_deref().copied().unwrap_or_default());
}

/// What a shadow view draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum ShadowViewRole {
    /// Every shadow caster, into the shadow map lighting samples (Bevy's
    /// shadow views).
    #[default]
    Full,
    /// The static casters of a cached light, into its static layer, or into
    /// its shadow map when it has no static layer.
    Static,
    /// The movable casters of a cached light, over its shadow map.
    Movers,
    /// For the audit: every caster of a cached view, found anew, into a
    /// texture of the audit's (`ShadowCacheAuditPlugin`), with the view's
    /// exact projection.
    Reference,
}

impl ShadowViewRole {
    /// The bits of a [`RetainedViewEntity::subview_index`] above the face or
    /// cascade index that hold the role; [`ShadowViewRole::Full`] is 0, so
    /// Bevy's own keys don't change.
    const SHIFT: u32 = 24;

    /// The subview index of this role's view of a face or cascade.
    pub fn subview_index(self, subview: u32) -> u32 {
        let role = match self {
            ShadowViewRole::Full => 0,
            ShadowViewRole::Static => 1,
            ShadowViewRole::Movers => 2,
            ShadowViewRole::Reference => 3,
        };
        subview | (role << Self::SHIFT)
    }

    /// The role of a view, from its subview index.
    pub fn of_subview_index(subview_index: u32) -> ShadowViewRole {
        match subview_index >> Self::SHIFT {
            1 => ShadowViewRole::Static,
            2 => ShadowViewRole::Movers,
            3 => ShadowViewRole::Reference,
            _ => ShadowViewRole::Full,
        }
    }

    /// Whether a view of this role draws a caster whose shadow is (or isn't)
    /// static.
    pub fn takes(self, static_shadow_caster: bool) -> bool {
        match self {
            ShadowViewRole::Full | ShadowViewRole::Reference => true,
            ShadowViewRole::Static => static_shadow_caster,
            ShadowViewRole::Movers => !static_shadow_caster,
        }
    }

    /// The order of its view among the views that aren't cameras: static
    /// views first, then the full ones, then the movers views.
    fn order(self) -> RootNonCameraViewOrder {
        RootNonCameraViewOrder(match self {
            ShadowViewRole::Static => -1,
            ShadowViewRole::Full | ShadowViewRole::Reference => 0,
            ShadowViewRole::Movers => 1,
        })
    }
}

/// Where a shadow view draws: a texture, a layer and a rectangle of it.
///
/// Today every spot light draws a whole layer of the shadow map size; a
/// region is the hook for a size per light (a layer of a size class, or a
/// rectangle of an atlas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShadowMapRegion {
    /// The texture its depth attachment is a view of.
    pub texture: TextureId,
    /// The array layer.
    pub layer: u32,
    /// The rectangle's corner, in texels.
    pub origin: UVec2,
    /// The rectangle's size, in texels.
    pub size: UVec2,
}

/// The key of a render view that draws a shadow map: the logical shadow view
/// (light, camera for cascades, face or cascade), what the render view draws
/// of it, and where. The role and the destination aren't part of the logical
/// identity the cache keeps its state by.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShadowViewKey {
    /// The logical shadow view.
    pub id: ShadowViewId,
    /// What the view draws.
    pub role: ShadowViewRole,
    /// Where it draws.
    pub region: ShadowMapRegion,
}

impl ShadowViewKey {
    /// The view's retained entity: the role is part of the subview index.
    pub fn retained_view_entity(&self) -> RetainedViewEntity {
        retained_shadow_view(&self.id, self.role)
    }
}

/// The retained entity of a role's render view of a shadow view: the light,
/// the camera (cascades), and the face or cascade with the role in the high
/// bits of the subview index ([`ShadowViewRole::subview_index`]).
pub(crate) fn retained_shadow_view(id: &ShadowViewId, role: ShadowViewRole) -> RetainedViewEntity {
    RetainedViewEntity::new(
        MainEntity::from(id.light),
        id.camera().map(MainEntity::from),
        role.subview_index(id.subview_index()),
    )
}

/// Set when a cached light's shadow view recorded its pass in this frame.
#[derive(Component, Clone, Default)]
#[doc(hidden)]
pub struct ShadowPassRecord(pub(crate) Arc<AtomicBool>);

impl ShadowPassRecord {
    /// Marks the pass as recorded.
    pub fn record(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    fn recorded(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// What the movers view of a cached shadow view restores from its static
/// layer before it draws.
#[derive(Component, Clone)]
#[doc(hidden)]
pub struct ShadowRestore {
    /// The quads to draw: instances of [`ShadowViewCaches`]'s quad buffer.
    pub(crate) quads: core::ops::Range<u32>,
    /// 4 to draw each quad as a triangle strip; 3 for the one triangle over
    /// the whole layer.
    pub(crate) vertices: u32,
    /// The static layer's bind group for the blit.
    pub(crate) bind_group: BindGroup,
    /// The view of the shadow map layer to restore into.
    pub(crate) destination_view: TextureView,
}

/// What the main world found for a cached shadow view this frame, from its
/// [`ShadowViewCache`].
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ExtractedShadowView {
    /// The view.
    pub(crate) id: ShadowViewId,
    /// The generation of its state in the main world.
    pub(crate) generation: u64,
    /// The revision of the view's static content.
    pub(crate) static_revision: u64,
    /// The tiles the movable casters touch.
    pub(crate) movers_tiles: ShadowTileMask,
    /// Whether the light reaches into a view; if not, the main world found no
    /// casters for it.
    pub(crate) lights_a_view: bool,
    /// The view has no usable static layer: all its casters are in one list,
    /// drawn by one render view ([`ShadowViewRole::Full`]), and its movers
    /// tiles are all set while a live caster is in its frustum.
    pub(crate) single_view: bool,
    /// The audit draws a reference of the view ([`ShadowViewRole::Reference`])
    /// and copies its map before another view can draw over it.
    pub(crate) reference: bool,
}

impl ExtractedShadowView {
    pub(crate) fn new(id: ShadowViewId, cache: &ShadowViewCache) -> Self {
        Self {
            id,
            generation: cache.generation(),
            static_revision: cache.static_revision(),
            movers_tiles: cache.movers_tiles().clone(),
            lights_a_view: cache.lights_a_view(),
            single_view: cache.single_view(),
            reference: cache.reference_entities().is_some(),
        }
    }
}

/// On a cached view's reference view ([`ShadowViewRole::Reference`]): after
/// it drew, the camera's shadow pass copies the view's map (its layer of the
/// shadow map texture) into the reference texture's layer 0, before another
/// view can draw over it.
#[derive(Component, Clone)]
#[doc(hidden)]
pub struct ShadowAuditCopy {
    pub(crate) source: Texture,
    pub(crate) source_layer: u32,
    pub(crate) destination: Texture,
}

/// A cached view's reference texture for the audit: layer 0 holds a copy of
/// its map, layer 1 its reference redraw.
struct ReferenceTarget {
    texture: Texture,
    /// Both layers, for the audit's compare.
    array_view: TextureView,
    /// Layer 1, the reference view's attachment.
    reference_view: TextureView,
    size: u32,
}

/// Extracted on a cached spot light's render entity: its one shadow view.
#[derive(Component, Clone, Debug)]
#[doc(hidden)]
pub struct ExtractedSpotShadowCache(pub(crate) ExtractedShadowView);

/// Extracted on a cached point light's render entity: its six faces' shadow
/// views, in the order of `CUBE_MAP_FACES`.
#[derive(Component, Clone, Debug)]
#[doc(hidden)]
pub struct ExtractedPointShadowCache(pub(crate) [ExtractedShadowView; 6]);

/// What a frame does for a cached shadow view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCachePlan {
    /// Nothing.
    Kept,
    /// Restore the tiles movable casters touch and were drawn into, draw the
    /// movable casters.
    Movers,
    /// Draw the static layer, restore it whole, draw the movable casters.
    Rebuild,
    /// Restore the whole static layer, draw the movable casters.
    Restore,
    /// Draw the static casters, then the movable ones, into the shadow map.
    Fallback,
    /// Draw nothing: the light lights no view.
    Dormant,
}

impl ShadowCachePlan {
    /// Plans a cached shadow view.
    ///
    /// - `has_layer`: the view has a static layer of the current size.
    /// - `static_valid`: the layer holds the current static revision.
    /// - `working_valid`: the view's shadow map is where it was last drawn
    ///   and holds the current static revision (with the movable casters of
    ///   `movers_last` on top).
    /// - `movers`: movable casters are in its frustum now, or were drawn into
    ///   its shadow map and not erased since.
    pub fn new(
        has_layer: bool,
        static_valid: bool,
        working_valid: bool,
        movers: bool,
    ) -> ShadowCachePlan {
        if !has_layer {
            return if working_valid && !movers {
                ShadowCachePlan::Kept
            } else {
                ShadowCachePlan::Fallback
            };
        }
        if !static_valid {
            ShadowCachePlan::Rebuild
        } else if !working_valid {
            ShadowCachePlan::Restore
        } else if movers {
            ShadowCachePlan::Movers
        } else {
            ShadowCachePlan::Kept
        }
    }

    /// Whether the static view draws.
    pub fn draws_static(&self) -> bool {
        matches!(self, ShadowCachePlan::Rebuild | ShadowCachePlan::Fallback)
    }

    /// Whether the movers view restores the whole static layer before it
    /// draws.
    pub fn restores_whole(&self) -> bool {
        matches!(self, ShadowCachePlan::Rebuild | ShadowCachePlan::Restore)
    }

    fn state(&self) -> ShadowCacheState {
        match self {
            ShadowCachePlan::Kept => ShadowCacheState::Kept,
            ShadowCachePlan::Movers => ShadowCacheState::Movers,
            ShadowCachePlan::Rebuild => ShadowCacheState::Rebuild,
            ShadowCachePlan::Restore => ShadowCacheState::Restore,
            ShadowCachePlan::Fallback => ShadowCacheState::Fallback,
            ShadowCachePlan::Dormant => ShadowCacheState::Dormant,
        }
    }
}

/// A cached view's static layer: a depth texture of its own.
struct StaticShadowLayer {
    texture: Texture,
    attachment_view: TextureView,
    /// Its depth, for the restore's blit and the audit.
    sampled_view: TextureView,
    bind_group: BindGroup,
    size: u32,
    bytes: u64,
}

/// Where a view's shadow map is drawn: the shadow map texture and layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WorkingDestination {
    texture: TextureId,
    layer: u32,
}

/// What the audit reads of a cached view ([`ShadowViewCaches::audit`]).
pub(crate) struct ShadowCacheAuditView {
    /// The shadow map texture it is drawn into this frame.
    pub texture: TextureId,
    /// Its layer of the shadow map texture.
    pub layer: u32,
    pub plan: ShadowCachePlan,
    /// The static revision the main world asked for.
    pub static_revision: u64,
    /// The revision the static layer holds, as acknowledged before this frame.
    pub static_content: Option<u64>,
    /// The revision the shadow map holds, as acknowledged before this frame.
    pub working: Option<u64>,
    /// Whether the static view recorded its pass this frame.
    pub static_drawn: bool,
    /// The texels of the tiles movable casters touch now.
    pub movers_texels: u64,
    /// The texels of the tiles movable casters were drawn into and that
    /// weren't restored since, before this frame.
    pub written_texels: u64,
    pub restored_texels: u64,
    /// The static layer's depth.
    pub static_layer: Option<TextureView>,
}

/// What this frame planned for a view, for [`acknowledge_shadow_caches`].
struct FramePlan {
    plan: ShadowCachePlan,
    /// Whether the view has a static layer and the blit to restore it.
    layer_usable: bool,
    static_revision: u64,
    destination: WorkingDestination,
    movers_now: ShadowTileMask,
    static_view: Option<(Entity, ShadowPassRecord)>,
    movers_view: Option<ShadowPassRecord>,
    /// The audit's reference view.
    reference_view: Option<ShadowPassRecord>,
    restored_texels: u64,
}

/// From this share of a map's tiles set on, a movers restore draws the whole
/// static layer with one quad instead of a quad per run of tiles.
const WHOLE_RESTORE_COVERAGE: f32 = 0.875;

/// What the render world keeps for a cached shadow view.
struct ShadowViewCacheState {
    /// The main world's generation of the view's state: another one is
    /// another view.
    generation: u64,
    /// The light's render entity.
    render_light: Entity,
    static_layer: Option<StaticShadowLayer>,
    /// The static revision the static layer holds (acknowledged).
    static_content: Option<u64>,
    /// Where the view's shadow map was drawn and which static revision it
    /// holds, as acknowledged.
    working: Option<(WorkingDestination, u64)>,
    /// The view of the shadow map layer, kept while it stays there.
    working_view: Option<(WorkingDestination, TextureView)>,
    /// The tiles of the shadow map movable casters were drawn into and that
    /// weren't restored since: what a restore must cover besides the tiles
    /// they touch now.
    written: ShadowTileMask,
    /// Where the view's shadow map is drawn this frame, if it was planned.
    destination: Option<WorkingDestination>,
    frame: Option<FramePlan>,
    seen: bool,
    /// The last frame planned it dormant.
    dormant: bool,
    /// For the audit (`ExtractedShadowView::reference`).
    reference: Option<ReferenceTarget>,
}

/// Static layers a view gave up, kept (and counted in the budget) until the
/// GPU can no longer use them: the render thread runs at most two frames
/// ahead of the GPU (wgpu's default frame latency), so a layer dropped
/// [`Self::FRAMES`] frames after its last use is released. The budget thus
/// bounds what the static layers hold at any time, not only the layers in
/// use; a view that would exceed it falls back until the bytes are free.
struct RetiringLayers<T> {
    /// (frames left, bytes, the layer).
    layers: Vec<(u32, u64, T)>,
    bytes: u64,
}

impl<T> Default for RetiringLayers<T> {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            bytes: 0,
        }
    }
}

impl<T> RetiringLayers<T> {
    const FRAMES: u32 = 3;

    fn retire(&mut self, bytes: u64, layer: T) {
        self.bytes += bytes;
        self.layers.push((Self::FRAMES, bytes, layer));
    }

    /// A frame passed: drops the layers whose time is up.
    fn tick(&mut self) {
        let mut freed = 0;
        self.layers.retain_mut(|(frames, bytes, _)| {
            *frames -= 1;
            let keep = *frames > 0;
            if !keep {
                freed += *bytes;
            }
            keep
        });
        self.bytes -= freed;
    }
}

/// The render world's cached shadow views, by [`ShadowViewId`], the bytes
/// their static layers take, and this frame's restore quads.
#[derive(Resource)]
#[doc(hidden)]
pub struct ShadowViewCaches {
    views: HashMap<ShadowViewId, ShadowViewCacheState>,
    /// The views extracted this frame, with their generations.
    extracted: HashMap<ShadowViewId, u64>,
    /// The bytes of the static layers in use.
    allocated_bytes: u64,
    /// Static layers given up, until the GPU released them.
    retiring: RetiringLayers<StaticShadowLayer>,
    /// The rectangles every restore of the frame draws, an instance each, in
    /// normalized device coordinates (min x, min y, max x, max y).
    restore_quads: RawBufferVec<[f32; 4]>,
    /// The view that last drew into each destination shared between views
    /// (the directional cascades' layers, which every camera draws its own
    /// cascades into in turn), as planned in render order; `None`: a view
    /// that isn't cached.
    owners: HashMap<WorkingDestination, Option<ShadowViewId>>,
}

impl Default for ShadowViewCaches {
    fn default() -> Self {
        Self {
            views: default(),
            extracted: default(),
            allocated_bytes: 0,
            retiring: default(),
            restore_quads: RawBufferVec::new(BufferUsages::VERTEX),
            owners: default(),
        }
    }
}

/// What [`ShadowViewCaches::prepare_spot_light`] needs.
pub(crate) struct PrepareCachedSpotLight<'a> {
    pub light_entity: Entity,
    pub light: &'a ExtractedPointLight,
    pub light_render_layers: &'a RenderLayers,
    pub spot_light_frustum: Option<&'a Frustum>,
    pub light_index: usize,
    pub layer: u32,
    pub shadow_map: &'a CachedTexture,
    pub map_size: u32,
    pub gpu_preprocessing_mode: GpuPreprocessingMode,
    pub extracted: &'a ExtractedSpotShadowCache,
    pub visible_entities: Option<&'a RenderShadowMapVisibleEntities>,
    /// Meshes whose GPU data couldn't be updated this frame.
    pub stale_meshes: &'a MainEntityHashSet,
    /// The settled casters among the meshes culled on the GPU only.
    pub static_gpu_culled: &'a MainEntityHashSet,
    /// Faults to inject, for tests.
    pub faults: ShadowCacheFaults,
}

/// A cached shadow view as its light type's adapter describes it to
/// [`ShadowViewCaches::prepare_view`]: what the main world found for it, its
/// projection and culling, the light it belongs to and where it is drawn.
pub(crate) struct CachedShadowView<'a> {
    /// Its render views run on their own, before the cameras (spot lights),
    /// or in a camera's shadow pass, in the order of the camera's view
    /// lights (directional cascades).
    pub root: bool,
    /// Its destination is shared with other views (a directional cascade's
    /// layer): its map is current only if no other view drew there since.
    pub shared_destination: bool,
    /// The view's exact clip-from-world, if it has one of its own (Bevy's
    /// cascades); else it is derived from the two matrices below.
    pub clip_from_world: Option<Mat4>,
    /// What the main world found for the view.
    pub extracted: &'a ExtractedShadowView,
    /// The light's render entity.
    pub light_entity: Entity,
    /// The light the view's render views belong to.
    pub light_kind: LightEntity,
    /// The render layers its render views draw.
    pub render_layers: &'a RenderLayers,
    /// The view's world-from-view transform, as its shadow view renders.
    pub world_from_view: GlobalTransform,
    /// The view's projection, as its shadow view renders.
    pub clip_from_view: Mat4,
    /// The frustum its casters are culled against.
    pub frustum: Option<&'a Frustum>,
    /// The name of its render views' passes; the role's suffix is added.
    pub pass_name: &'a str,
    /// The shadow map texture it is drawn into, its layer and size.
    pub shadow_map: &'a CachedTexture,
    pub layer: u32,
    pub map_size: u32,
    pub gpu_preprocessing_mode: GpuPreprocessingMode,
    /// The visible entities of the light's render views.
    pub visible_entities: Option<&'a RenderShadowMapVisibleEntities>,
    /// Meshes whose GPU data couldn't be updated this frame.
    pub stale_meshes: &'a MainEntityHashSet,
    /// The settled casters among the meshes culled on the GPU only.
    pub static_gpu_culled: &'a MainEntityHashSet,
    /// Faults to inject, for tests.
    pub faults: ShadowCacheFaults,
}

impl ShadowViewCaches {
    /// Starts a frame: forgets views that are no longer cached (or whose
    /// state the main world created anew) and frees static layers beyond the
    /// budget. `extracted`: the cached views the main world extracted.
    pub(crate) fn prepare_for_new_frame<'a>(
        &mut self,
        extracted: impl Iterator<Item = &'a ExtractedShadowView>,
        settings: &ShadowCacheSettings,
    ) {
        self.retiring.tick();
        self.extracted.clear();
        self.extracted
            .extend(extracted.map(|view| (view.id, view.generation)));
        let retiring = &mut self.retiring;
        let extracted = &self.extracted;
        let mut freed = 0;
        self.views.retain(|id, state| {
            let keep = extracted.get(id) == Some(&state.generation);
            if !keep && let Some(layer) = state.static_layer.take() {
                freed += layer.bytes;
                retiring.retire(layer.bytes, layer);
            }
            keep
        });
        self.allocated_bytes -= freed;

        // A smaller budget: free the layers of the last views, in the order
        // of their lights' render entities. The main world redraws every
        // view when the settings change.
        if self.allocated_bytes > settings.budget_bytes {
            let mut views: Vec<_> = self
                .views
                .iter()
                .map(|(id, state)| (state.render_light, *id))
                .collect();
            views.sort_unstable();
            for (_, id) in views.into_iter().rev() {
                if self.allocated_bytes <= settings.budget_bytes {
                    break;
                }
                let state = self.views.get_mut(&id).unwrap();
                if let Some(layer) = state.static_layer.take() {
                    self.allocated_bytes -= layer.bytes;
                    state.static_content = None;
                    self.retiring.retire(layer.bytes, layer);
                }
            }
        }

        for state in self.views.values_mut() {
            state.seen = false;
            state.frame = None;
            state.destination = None;
        }
        self.restore_quads.clear();
    }

    /// The cached views planned this frame: (view, its light's render
    /// entity).
    pub(crate) fn audited_views(&self) -> impl Iterator<Item = (ShadowViewId, Entity)> + '_ {
        self.views
            .iter()
            .filter(|(_, state)| state.frame.is_some())
            .map(|(id, state)| (*id, state.render_light))
    }

    /// What [`ShadowCacheAuditPlugin`](crate::ShadowCacheAuditPlugin) reads
    /// of a cached view that was planned this frame, after the frame's passes
    /// and before they are acknowledged.
    pub(crate) fn audit(&self, view: &ShadowViewId) -> Option<ShadowCacheAuditView> {
        let state = self.views.get(view)?;
        let frame = state.frame.as_ref()?;
        Some(ShadowCacheAuditView {
            texture: frame.destination.texture,
            layer: frame.destination.layer,
            plan: frame.plan,
            static_revision: frame.static_revision,
            static_content: state.static_content,
            working: state.working.map(|(_, revision)| revision),
            static_drawn: frame
                .static_view
                .as_ref()
                .is_some_and(|(_, record)| record.recorded()),
            movers_texels: frame.movers_now.texels(),
            written_texels: state.written.texels(),
            restored_texels: frame.restored_texels,
            static_layer: state
                .static_layer
                .as_ref()
                .map(|layer| layer.sampled_view.clone()),
        })
    }

    /// The audit's reference of a view planned this frame: its texture's
    /// layers (0: the view's map, 1: the reference), its size, and whether
    /// the reference view recorded its pass.
    pub(crate) fn reference(&self, view: &ShadowViewId) -> Option<(TextureView, u32, bool)> {
        let state = self.views.get(view)?;
        let frame = state.frame.as_ref()?;
        let target = state.reference.as_ref()?;
        Some((
            target.array_view.clone(),
            target.size,
            frame
                .reference_view
                .as_ref()
                .is_some_and(ShadowPassRecord::recorded),
        ))
    }

    /// The spot light adapter: describes a cached spot light's one shadow
    /// view by its projection and plans it ([`Self::prepare_view`]).
    pub(crate) fn prepare_spot_light(
        &mut self,
        commands: &mut Commands,
        args: PrepareCachedSpotLight,
        view_entities: &mut PointAndSpotLightViewEntities,
        settings: &ShadowCacheSettings,
        pipelines: (
            &RenderDevice,
            Option<&ShadowRestorePipeline>,
            &PipelineCache,
        ),
        shadow_render_phases: &mut ViewBinnedRenderPhases<Shadow>,
        live_shadow_mapping_lights: &mut HashSet<RetainedViewEntity>,
    ) {
        let spot_world_from_view = spot_light_world_from_view(&args.light.transform);
        let angle = args
            .light
            .spot_light_angles
            .expect("a cached light should be a spot light")
            .1;
        let spot_projection = spot_light_clip_from_view(angle, args.light.shadow_map_near_z);
        let pass_name = format!("shadow_spot_light_{}", args.light_index);
        self.prepare_view(
            commands,
            CachedShadowView {
                root: true,
                shared_destination: false,
                clip_from_world: None,
                extracted: &args.extracted.0,
                light_entity: args.light_entity,
                light_kind: LightEntity::Spot {
                    light_entity: args.light_entity,
                },
                render_layers: args.light_render_layers,
                world_from_view: GlobalTransform::from(spot_world_from_view),
                clip_from_view: spot_projection,
                frustum: args.spot_light_frustum,
                pass_name: &pass_name,
                shadow_map: args.shadow_map,
                layer: args.layer,
                map_size: args.map_size,
                gpu_preprocessing_mode: args.gpu_preprocessing_mode,
                visible_entities: args.visible_entities,
                stale_meshes: args.stale_meshes,
                static_gpu_culled: args.static_gpu_culled,
                faults: args.faults,
            },
            &mut view_entities.0,
            settings,
            pipelines,
            shadow_render_phases,
            live_shadow_mapping_lights,
        );
    }

    /// Plans a cached shadow view and sets up its render views (at most two,
    /// in `view_entities`: the static view's, the movers view's).
    pub(crate) fn prepare_view(
        &mut self,
        commands: &mut Commands,
        args: CachedShadowView,
        view_entities: &mut Vec<Entity>,
        settings: &ShadowCacheSettings,
        (render_device, restore_pipeline, pipeline_cache): (
            &RenderDevice,
            Option<&ShadowRestorePipeline>,
            &PipelineCache,
        ),
        shadow_render_phases: &mut ViewBinnedRenderPhases<Shadow>,
        live_shadow_mapping_lights: &mut HashSet<RetainedViewEntity>,
    ) {
        let size = args.map_size;
        let id = args.extracted.id;
        let state = self
            .views
            .entry(id)
            .or_insert_with(|| ShadowViewCacheState {
                generation: args.extracted.generation,
                render_light: args.light_entity,
                static_layer: None,
                static_content: None,
                working: None,
                working_view: None,
                written: ShadowTileMask::new(args.map_size),
                destination: None,
                frame: None,
                seen: false,
                dormant: false,
                reference: None,
            });
        state.seen = true;
        state.render_light = args.light_entity;

        // A static layer of another size is drawn again at the new size.
        if state
            .static_layer
            .as_ref()
            .is_some_and(|layer| layer.size != size)
            && let Some(layer) = state.static_layer.take()
        {
            self.allocated_bytes -= layer.bytes;
            state.static_content = None;
            self.retiring.retire(layer.bytes, layer);
        }
        // Stable slots: a view keeps its layer while it's cached; views get
        // one in Bevy's light order while the budget lasts.
        let layer_bytes = size as u64 * size as u64 * 4;
        if state.static_layer.is_none()
            && self.allocated_bytes + self.retiring.bytes + layer_bytes <= settings.budget_bytes
            && let Some(restore_pipeline) = restore_pipeline
        {
            state.static_layer = Some(create_static_layer(
                render_device,
                restore_pipeline,
                pipeline_cache,
                size,
                layer_bytes,
            ));
            state.static_content = None;
            self.allocated_bytes += layer_bytes;
        }
        let has_layer = state.static_layer.is_some();
        // A static layer is of use only once the blit that restores it is
        // compiled; until then the view falls back to drawing everything.
        let restore_ready = !args.faults.restore_not_ready
            && restore_pipeline.is_some_and(|restore_pipeline| {
                pipeline_cache
                    .get_render_pipeline(restore_pipeline.pipeline_id)
                    .is_some()
            });

        // Its render views need their lists of visible entities; a view
        // without them this frame draws nothing and loses its place.
        let single_view = args.extracted.single_view;
        let roles: &[ShadowViewRole] = if single_view {
            &[ShadowViewRole::Full]
        } else {
            &[ShadowViewRole::Static, ShadowViewRole::Movers]
        };
        let has_lists = args.visible_entities.is_some_and(|visible_entities| {
            roles.iter().all(|role| {
                visible_entities
                    .subviews
                    .contains_key(&retained_shadow_view(&id, *role))
            })
        });
        if !has_lists {
            despawn_view_entities(commands, core::mem::take(view_entities));
            state.seen = false;
            return;
        }

        // Movable casters culled on the GPU only have no CPU bounds; one whose
        // GPU data wasn't updated this frame is drawn where it was, not where
        // its bounds are. Either can be anywhere. (A single view's list holds
        // the settled casters too.)
        let unbounded_movers = args
            .visible_entities
            .and_then(|visible_entities| {
                visible_entities
                    .subviews
                    .get(&retained_shadow_view(&id, roles[roles.len() - 1]))
            })
            .and_then(|(_, entities)| entities.get::<Mesh3d>())
            .is_some_and(|entities| {
                (if single_view {
                    entities
                        .entities_gpu_culling
                        .keys()
                        .any(|main_entity| !args.static_gpu_culled.contains(main_entity))
                } else {
                    !entities.entities_gpu_culling.is_empty()
                }) || (!args.stale_meshes.is_empty()
                    && entities
                        .entities_cpu_culling
                        .iter()
                        .any(|(_, main_entity)| args.stale_meshes.contains(main_entity)))
            });
        let mut movers_now = args.extracted.movers_tiles.clone();
        if movers_now.size() != size {
            let any = !movers_now.is_empty();
            movers_now.reset(size);
            if any {
                movers_now.fill();
            }
        }
        if unbounded_movers {
            movers_now.fill();
        }

        let destination = WorkingDestination {
            texture: args.shadow_map.texture.id(),
            layer: args.layer,
        };
        let revision = args.extracted.static_revision;
        // A view drawn into another place now (a point light whose index
        // changed, a resized shadow map array) no longer has its map there,
        // even if it comes back to its old place later (another view may
        // have drawn there since): checked every frame, dormant ones too.
        // The fault takes the old map for current.
        if state
            .working
            .is_some_and(|(working, _)| working != destination)
            && !args.faults.miss_index_change
        {
            state.working = None;
        }
        let static_valid = has_layer && state.static_content == Some(revision);
        // A shared destination: another view may have drawn there since (a
        // camera drawing its own cascades into the same layer).
        let owned = !args.shared_destination
            || (!args.faults.miss_ownership && self.owners.get(&destination) == Some(&Some(id)))
            || (args.faults.miss_ownership && state.working.is_some());
        let working_valid = owned
            && (state.working == Some((destination, revision))
                || (args.faults.miss_index_change
                    && state.working.is_some_and(|(_, working)| working == revision)));
        // A view whose light lights no camera's view draws nothing until it
        // lights one again; the fault forgets in that frame what was written
        // before.
        let dormant = !args.extracted.lights_a_view;
        let drop_written = args.faults.drop_written
            || (args.faults.skip_reentry_restore && state.dormant && !dormant);
        state.dormant = dormant;
        let written = !drop_written && !state.written.is_empty();
        // A single view draws everything: planned as without a layer.
        let layer_usable = has_layer && restore_ready;
        let plan = if dormant {
            ShadowCachePlan::Dormant
        } else {
            ShadowCachePlan::new(
                layer_usable && !single_view,
                static_valid,
                working_valid,
                !movers_now.is_empty() || written,
            )
        };
        state.destination = Some(destination);
        if args.shared_destination {
            self.owners.insert(destination, Some(id));
        }

        // The render views this plan draws; the others go. (A third for the
        // audit's reference.)
        let slots = if args.extracted.reference { 3 } else { 2 };
        if view_entities.len() != slots {
            despawn_view_entities(commands, core::mem::take(view_entities));
            *view_entities = vec![Entity::PLACEHOLDER; slots];
        }
        let draws_static = plan.draws_static();
        let draws_movers = match plan {
            ShadowCachePlan::Kept | ShadowCachePlan::Dormant => false,
            ShadowCachePlan::Fallback => !single_view && !movers_now.is_empty(),
            _ => true,
        };
        for (index, needed) in [draws_static, draws_movers].into_iter().enumerate() {
            let entity = &mut view_entities[index];
            if needed && *entity == Entity::PLACEHOLDER {
                *entity = commands.spawn_empty().id();
            } else if !needed && *entity != Entity::PLACEHOLDER {
                despawn_view_entities(commands, vec![*entity]);
                *entity = Entity::PLACEHOLDER;
            }
        }

        let working_view = match &state.working_view {
            Some((working_destination, view)) if *working_destination == destination => {
                view.clone()
            }
            _ => {
                let view = args.shadow_map.texture.create_view(&TextureViewDescriptor {
                    label: Some("cached_shadow_map_texture_view"),
                    format: None,
                    dimension: Some(TextureViewDimension::D2),
                    usage: None,
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: args.layer,
                    array_layer_count: Some(1u32),
                });
                state.working_view = Some((destination, view.clone()));
                view
            }
        };

        let region = ShadowMapRegion {
            texture: destination.texture,
            layer: args.layer,
            origin: UVec2::ZERO,
            size: UVec2::splat(size),
        };

        let mut frame = FramePlan {
            plan,
            layer_usable,
            static_revision: revision,
            destination,
            movers_now: ShadowTileMask::default(),
            static_view: None,
            movers_view: None,
            reference_view: None,
            restored_texels: 0,
        };

        let mut spawn_view = |commands: &mut Commands,
                              entity: Entity,
                              role: ShadowViewRole,
                              region: ShadowMapRegion,
                              depth_attachment: DepthStencilViewAttachment|
         -> ShadowPassRecord {
            let key = ShadowViewKey { id, role, region };
            let record = ShadowPassRecord::default();
            let role_name = match role {
                ShadowViewRole::Full => "",
                ShadowViewRole::Static => "_static",
                ShadowViewRole::Movers => "_movers",
                ShadowViewRole::Reference => "_reference",
            };
            commands.entity(entity).insert((
                ShadowView {
                    depth_attachment,
                    pass_name: format!("{}{role_name}", args.pass_name),
                },
                ExtractedView {
                    retained_view_entity: key.retained_view_entity(),
                    viewport: UVec4::new(
                        region.origin.x,
                        region.origin.y,
                        region.size.x,
                        region.size.y,
                    ),
                    world_from_view: args.world_from_view,
                    clip_from_view: args.clip_from_view,
                    clip_from_world: args.clip_from_world,
                    target_format: CORE_3D_SHADOW_MAP_FORMAT,
                    color_grading: Default::default(),
                    invert_culling: false,
                },
                *args.frustum.unwrap(),
                args.light_kind,
                args.render_layers.clone(),
                key,
                record.clone(),
            ));
            if args.root {
                commands
                    .entity(entity)
                    .insert((RootNonCameraView(Core3d.intern()), role.order()));
            }
            if !matches!(args.gpu_preprocessing_mode, GpuPreprocessingMode::Culling) {
                commands.entity(entity).insert(NoIndirectDrawing);
            }
            shadow_render_phases
                .prepare_for_new_frame(key.retained_view_entity(), args.gpu_preprocessing_mode);
            live_shadow_mapping_lights.insert(key.retained_view_entity());
            record
        };

        if draws_static {
            // Into the static layer, or straight into the shadow map when
            // falling back; both cleared first.
            let (view, view_region) = match (&state.static_layer, plan) {
                (Some(layer), ShadowCachePlan::Rebuild) => (
                    layer.attachment_view.clone(),
                    ShadowMapRegion {
                        texture: layer.texture.id(),
                        layer: 0,
                        ..region
                    },
                ),
                _ => (working_view.clone(), region),
            };
            let attachment = DepthStencilViewAttachment::new(
                DepthStencilViews::DepthOnly { depth_view: view },
                Some(0.0),
                None,
            );
            let entity = view_entities[0];
            let role = if single_view {
                ShadowViewRole::Full
            } else {
                ShadowViewRole::Static
            };
            let record = spawn_view(commands, entity, role, view_region, attachment);
            frame.static_view = Some((entity, record));
        }
        if draws_movers {
            // Over what the shadow map holds or what the restore wrote: never
            // cleared.
            let attachment = DepthStencilViewAttachment::new(
                DepthStencilViews::DepthOnly {
                    depth_view: working_view.clone(),
                },
                None,
                None,
            );
            let entity = view_entities[1];
            frame.movers_view = Some(spawn_view(
                commands,
                entity,
                ShadowViewRole::Movers,
                region,
                attachment,
            ));
            let restores = matches!(plan, ShadowCachePlan::Movers) || plan.restores_whole();
            match &state.static_layer {
                Some(layer) if restores && !args.faults.skip_restore => {
                    // One quad per run of tiles in a row, or one triangle
                    // over the whole layer, in normalized device coordinates.
                    let first = self.restore_quads.len() as u32;
                    let quads = &mut self.restore_quads;
                    let ndc_per_texel = 2.0 / size as f32;
                    let mut push = |rect: URect| {
                        let to_ndc = |texels: u32| texels as f32 * ndc_per_texel;
                        quads.push([
                            to_ndc(rect.min.x) - 1.0,
                            1.0 - to_ndc(rect.max.y),
                            to_ndc(rect.max.x) - 1.0,
                            1.0 - to_ndc(rect.min.y),
                        ]);
                    };
                    let mut region = None;
                    if !plan.restores_whole() {
                        let mut tiles = movers_now.clone();
                        if !drop_written {
                            tiles.union_with(&state.written);
                        }
                        if args.faults.drop_one_tile {
                            tiles.drop_one_tile(&movers_now);
                        }
                        // Most of the map: one quad is faster than a quad
                        // per run (measured on both GPUs, 2048 and 4096).
                        if tiles.coverage() < WHOLE_RESTORE_COVERAGE {
                            region = Some(tiles);
                        }
                    }
                    let vertices = match region {
                        Some(tiles) => {
                            tiles.for_each_run(|rect| {
                                frame.restored_texels += rect.width() as u64 * rect.height() as u64;
                                push(rect);
                            });
                            4
                        }
                        None => {
                            // The strip's first triangle over a rectangle of
                            // twice the map's size: one triangle covering it,
                            // like Bevy's fullscreen triangle (a quad's two
                            // triangles were measured up to 5 % slower).
                            self.restore_quads.push([-1.0, -1.0, 3.0, 3.0]);
                            frame.restored_texels = size as u64 * size as u64;
                            3
                        }
                    };
                    let destination_view = if args.faults.wrong_layer {
                        let layers = args.shadow_map.texture.depth_or_array_layers();
                        let wrong = if args.layer + 1 < layers {
                            args.layer + 1
                        } else {
                            args.layer.saturating_sub(1)
                        };
                        args.shadow_map.texture.create_view(&TextureViewDescriptor {
                            label: Some("spot_light_shadow_map_texture_view"),
                            dimension: Some(TextureViewDimension::D2),
                            base_array_layer: wrong,
                            array_layer_count: Some(1u32),
                            ..default()
                        })
                    } else {
                        working_view.clone()
                    };
                    commands.entity(entity).insert(ShadowRestore {
                        quads: first..self.restore_quads.len() as u32,
                        vertices,
                        bind_group: layer.bind_group.clone(),
                        destination_view,
                    });
                }
                _ => {
                    commands.entity(entity).remove::<ShadowRestore>();
                }
            }
        }
        // The audit's reference: every caster, with the same projection,
        // into a texture of its own; then the view's map is copied beside it.
        if args.extracted.reference {
            if state
                .reference
                .as_ref()
                .is_none_or(|target| target.size != size)
            {
                let texture = render_device.create_texture(&TextureDescriptor {
                    label: Some("shadow_cache_audit_reference"),
                    size: Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 2,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: CORE_3D_SHADOW_MAP_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let array_view = texture.create_view(&TextureViewDescriptor {
                    label: Some("shadow_cache_audit_reference_array"),
                    dimension: Some(TextureViewDimension::D2Array),
                    aspect: TextureAspect::DepthOnly,
                    ..default()
                });
                let reference_view = texture.create_view(&TextureViewDescriptor {
                    label: Some("shadow_cache_audit_reference_layer"),
                    dimension: Some(TextureViewDimension::D2),
                    base_array_layer: 1,
                    array_layer_count: Some(1),
                    ..default()
                });
                state.reference = Some(ReferenceTarget {
                    texture,
                    array_view,
                    reference_view,
                    size,
                });
            }
            let target = state.reference.as_ref().unwrap();
            let entity = &mut view_entities[2];
            if *entity == Entity::PLACEHOLDER {
                *entity = commands.spawn_empty().id();
            }
            let entity = *entity;
            let attachment = DepthStencilViewAttachment::new(
                DepthStencilViews::DepthOnly {
                    depth_view: target.reference_view.clone(),
                },
                Some(0.0),
                None,
            );
            let reference_region = ShadowMapRegion {
                texture: target.texture.id(),
                layer: 1,
                ..region
            };
            frame.reference_view = Some(spawn_view(
                commands,
                entity,
                ShadowViewRole::Reference,
                reference_region,
                attachment,
            ));
            commands.entity(entity).insert(ShadowAuditCopy {
                source: args.shadow_map.texture.clone(),
                source_layer: args.layer,
                destination: target.texture.clone(),
            });
        } else {
            state.reference = None;
        }
        frame.movers_now = movers_now;
        state.frame = Some(frame);
    }

    /// Notes that a view that isn't cached draws into a shared destination
    /// (a directional cascade's layer) this frame, after the views planned so
    /// far: a cached view's map there is no longer current.
    pub(crate) fn note_shared_draw(&mut self, texture: TextureId, layer: u32) {
        self.owners
            .insert(WorkingDestination { texture, layer }, None);
    }

    /// Ends planning: a view that wasn't planned this frame (its light
    /// hidden, out of view) no longer has a shadow map where it was drawn.
    /// Uploads the frame's restore quads.
    pub(crate) fn finish_frame(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        for state in self.views.values_mut() {
            if !state.seen {
                state.working = None;
                state.working_view = None;
            }
        }
        if !self.restore_quads.is_empty() {
            self.restore_quads.write_buffer(render_device, render_queue);
        }
    }
}

fn despawn_view_entities(commands: &mut Commands, entities: Vec<Entity>) {
    let entities: Vec<_> = entities
        .into_iter()
        .filter(|entity| *entity != Entity::PLACEHOLDER)
        .collect();
    if entities.is_empty() {
        return;
    }
    commands.queue(move |world: &mut World| {
        for entity in entities {
            world.despawn(entity);
        }
    });
}

fn create_static_layer(
    render_device: &RenderDevice,
    restore_pipeline: &ShadowRestorePipeline,
    pipeline_cache: &PipelineCache,
    size: u32,
    bytes: u64,
) -> StaticShadowLayer {
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("spot_light_static_shadow_layer"),
        size: Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: CORE_3D_SHADOW_MAP_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let attachment_view = texture.create_view(&TextureViewDescriptor {
        label: Some("spot_light_static_shadow_layer_view"),
        ..default()
    });
    let sampled_view = texture.create_view(&TextureViewDescriptor {
        label: Some("spot_light_static_shadow_layer_depth_view"),
        aspect: TextureAspect::DepthOnly,
        ..default()
    });
    let bind_group = render_device.create_bind_group(
        "spot_light_static_shadow_layer_bind_group",
        &pipeline_cache.get_bind_group_layout(&restore_pipeline.layout),
        &BindGroupEntries::single(&sampled_view),
    );
    StaticShadowLayer {
        texture,
        attachment_view,
        sampled_view,
        bind_group,
        size,
        bytes,
    }
}

/// The blit that restores rectangles of a static layer into a shadow map.
#[derive(Resource)]
#[doc(hidden)]
pub struct ShadowRestorePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedRenderPipelineId,
}

fn init_shadow_restore_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "shadow_restore_bind_group_layout",
        &BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, texture_depth_2d()),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "shadow_restore.wesl");
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("shadow_restore_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            // A rectangle per instance.
            buffers: vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                [VertexFormat::Float32x4],
            )],
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            ..default()
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_SHADOW_MAP_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    });
    commands.insert_resource(ShadowRestorePipeline {
        layout,
        pipeline_id,
    });
}

/// Restores rectangles of a cached view's static layer, or all of it, into
/// its shadow map before its movers view draws: one instanced quad per
/// rectangle writing each texel's depth. [`ShadowViewCaches::prepare_view`]
/// plans a restore only once the blit's pipeline is compiled.
pub(crate) fn restore_shadow_map(world: &World, restore: &ShadowRestore, ctx: &mut RenderContext) {
    let Some(quads) = world
        .resource::<ShadowViewCaches>()
        .restore_quads
        .buffer()
        .filter(|_| !restore.quads.is_empty())
    else {
        return;
    };
    let pipeline_cache = world.resource::<PipelineCache>();
    let Some(pipeline) =
        world
            .get_resource::<ShadowRestorePipeline>()
            .and_then(|restore_pipeline| {
                pipeline_cache.get_render_pipeline(restore_pipeline.pipeline_id)
            })
    else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("shadow_restore"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &restore.destination_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "shadow_restore");
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &restore.bind_group, &[]);
    render_pass.set_vertex_buffer(0, quads.slice(..));
    render_pass.draw(0..restore.vertices, restore.quads.clone());
    pass_span.end(&mut render_pass);
}

/// For the audit: copies a cached view's map beside its reference
/// ([`ShadowAuditCopy`]).
pub(crate) fn copy_for_audit(copy: &ShadowAuditCopy, ctx: &mut RenderContext) {
    ctx.command_encoder().copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: &copy.source,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: 0,
                z: copy.source_layer,
            },
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: &copy.destination,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: copy.destination.width(),
            height: copy.destination.height(),
            depth_or_array_layers: 1,
        },
    );
}

/// What [`acknowledge_shadow_caches`] checks a static view's casters
/// against.
#[derive(SystemParam)]
pub(crate) struct ShadowViewReadiness<'w, 's> {
    views: Query<
        'w,
        's,
        (
            &'static ExtractedView,
            &'static LightEntity,
            &'static RenderLayers,
            Has<PreprocessBindGroups>,
        ),
    >,
    visible_entities: Query<'w, 's, &'static RenderShadowMapVisibleEntities>,
    render_meshes: Res<'w, RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<'w, RenderMeshInstances>,
    render_materials: Res<'w, ErasedRenderAssets<PreparedMaterial>>,
    render_material_instances: Res<'w, RenderMaterialInstances>,
    mesh_allocator: Res<'w, MeshAllocator>,
    pipelines: Res<'w, SpecializedShadowMaterialPipelineCache>,
    pipeline_cache: Res<'w, PipelineCache>,
    pending_shadow_queues: Res<'w, PendingShadowQueues>,
    gpu_preprocessing_support: Res<'w, GpuPreprocessingSupport>,
    preprocess_pipelines: Option<Res<'w, PreprocessPipelines>>,
    faults: Option<Res<'w, ShadowCacheFaults>>,
}

/// Whether every GPU preprocessing pipeline a shadow view's draws go through
/// is compiled: a view drawn before one of them is drew nothing.
fn preprocess_pipelines_are_loaded(
    pipelines: &PreprocessPipelines,
    pipeline_cache: &PipelineCache,
    gpu_preprocessing_support: &GpuPreprocessingSupport,
) -> bool {
    let loaded = |id: Option<CachedComputePipelineId>| {
        id.is_some_and(|id| pipeline_cache.get_compute_pipeline(id).is_some())
    };
    pipelines.pipelines_are_loaded(pipeline_cache, gpu_preprocessing_support)
        && loaded(pipelines.bin_unpacking.pipeline_id)
        && loaded(
            pipelines
                .uniform_allocation
                .local_scan
                .pipeline_id_local_scan,
        )
        && loaded(
            pipelines
                .uniform_allocation
                .global_scan
                .pipeline_id_global_scan,
        )
        && loaded(pipelines.uniform_allocation.fan.pipeline_id_fan)
}

impl ShadowViewReadiness<'_, '_> {
    /// Whether every caster of a shadow view was drawn: each one either
    /// doesn't cast a shadow in this view, or had its mesh, material and
    /// pipeline ready. Call after the frame's passes, when the pipeline cache
    /// is as the passes found it.
    fn is_ready(&self, view_entity: Entity) -> bool {
        if self
            .faults
            .as_deref()
            .is_some_and(|faults| faults.acknowledge_early)
        {
            return true;
        }
        let Ok((view, light_entity, view_layers, has_preprocess_bind_groups)) =
            self.views.get(view_entity)
        else {
            return false;
        };
        let retained_view_entity = view.retained_view_entity;
        // The visible entities the view's pass drew, as it looked them up.
        let (light_entity, visible_entities_key) =
            shadow_map_visible_entities_key(light_entity, view);
        if self.gpu_preprocessing_support.max_supported_mode != GpuPreprocessingMode::None
            && (!has_preprocess_bind_groups
                || !self
                    .preprocess_pipelines
                    .as_deref()
                    .is_some_and(|pipelines| {
                        preprocess_pipelines_are_loaded(
                            pipelines,
                            &self.pipeline_cache,
                            &self.gpu_preprocessing_support,
                        )
                    }))
        {
            return false;
        }
        if self
            .pending_shadow_queues
            .get(&retained_view_entity)
            .is_some_and(|pending| !pending.current_frame.is_empty())
        {
            return false;
        }
        let Some(visible_entities) = self
            .visible_entities
            .get(light_entity)
            .ok()
            .and_then(|visible_entities| visible_entities.subviews.get(&visible_entities_key))
            .and_then(|(_, visible_entities)| visible_entities.get::<Mesh3d>())
        else {
            // Nothing visible: nothing to wait for.
            return true;
        };
        let view_pipelines = self.pipelines.get(&retained_view_entity);
        visible_entities.iter_visible().all(|(_, main_entity)| {
            let Some(material_instance) = self.render_material_instances.instances.get(main_entity)
            else {
                return false;
            };
            let Some(mesh_instance) = self
                .render_mesh_instances
                .render_mesh_queue_data(*main_entity)
            else {
                return false;
            };
            let Some(material) = self.render_materials.get(material_instance.asset_id) else {
                return false;
            };
            if !material.properties.shadows_enabled
                || !mesh_instance
                    .flags()
                    .contains(RenderMeshInstanceFlags::SHADOW_CASTER)
                || !view_layers.intersects(mesh_instance.render_layers.as_ref().unwrap_or_default())
                || material.properties.prepass_specialize.is_none()
            {
                // Never drawn into a shadow map.
                return true;
            }
            if self
                .render_meshes
                .get(mesh_instance.mesh_asset_id())
                .is_none()
                || self
                    .mesh_allocator
                    .mesh_slabs(&mesh_instance.mesh_asset_id())
                    .is_none()
            {
                return false;
            }
            let Some(&(pipeline_id, ..)) =
                view_pipelines.and_then(|view_pipelines| view_pipelines.get(main_entity))
            else {
                return false;
            };
            match self.pipeline_cache.get_render_pipeline_state(pipeline_id) {
                // Compiling, or a shader still loading (retried).
                CachedPipelineState::Queued
                | CachedPipelineState::Creating(_)
                | CachedPipelineState::Err(
                    ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable,
                ) => false,
                // A pipeline that fails never draws, in a full redraw either.
                CachedPipelineState::Ok(_) | CachedPipelineState::Err(_) => true,
            }
        })
    }
}

/// The tiles movable casters were drawn into and not restored since, after a
/// frame that planned a movers pass over `movers_now`: the restore erased what
/// was written and the pass wrote `movers_now`. A pass that wasn't recorded
/// may still have left what was written before.
fn written_after_pass(written: &mut ShadowTileMask, movers_now: ShadowTileMask, recorded: bool) {
    if recorded {
        *written = movers_now;
    } else {
        written.union_with(&movers_now);
    }
}

/// After the frame's passes: counts a cached view's static content and
/// shadow map as valid only if the passes that drew them were recorded and
/// every caster of the static view was ready, and reports to the main world,
/// by view and generation: a report on another generation acknowledges
/// nothing.
pub(crate) fn acknowledge_shadow_caches(
    mut caches: ResMut<ShadowViewCaches>,
    feedback: Res<ShadowCacheFeedback>,
    readiness: ShadowViewReadiness,
    (mut reported, mut live, mut statuses): (
        Local<HashSet<ShadowViewId>>,
        Local<HashSet<ShadowViewId>>,
        Local<Vec<(ShadowViewId, ShadowCacheViewStatus)>>,
    ),
) {
    statuses.clear();
    for (id, state) in caches.views.iter_mut() {
        if let Some(frame) = state.frame.take() {
            // A Fallback frame with live casters in the map is followed by
            // another Fallback (their tiles stay written), so whether its
            // casters were ready never counts: the readiness check, a look at
            // every caster of the view, is skipped and nothing acknowledged.
            let counts = frame.plan != ShadowCachePlan::Fallback || frame.movers_now.is_empty();
            let static_drawn = counts
                && frame.static_view.as_ref().is_some_and(|(entity, record)| {
                    record.recorded() && readiness.is_ready(*entity)
                });
            let movers_drawn = frame
                .movers_view
                .as_ref()
                .is_none_or(ShadowPassRecord::recorded);
            let revision = frame.static_revision;
            let working = Some((frame.destination, revision));
            match frame.plan {
                ShadowCachePlan::Kept | ShadowCachePlan::Dormant => {}
                ShadowCachePlan::Rebuild => {
                    if static_drawn {
                        state.static_content = Some(revision);
                    }
                    state.working = working.filter(|_| static_drawn && movers_drawn);
                }
                ShadowCachePlan::Restore | ShadowCachePlan::Movers => {
                    state.working = working.filter(|_| movers_drawn);
                }
                ShadowCachePlan::Fallback => {
                    state.working = working.filter(|_| static_drawn && movers_drawn);
                }
            }
            if !matches!(frame.plan, ShadowCachePlan::Kept | ShadowCachePlan::Dormant) {
                written_after_pass(&mut state.written, frame.movers_now, movers_drawn);
            }
            let has_layer = state.static_layer.is_some();
            statuses.push((
                *id,
                ShadowCacheViewStatus {
                    generation: state.generation,
                    acknowledged_static_revision: if has_layer {
                        state.static_content
                    } else {
                        state.working.map(|(_, revision)| revision)
                    },
                    fallback: !frame.layer_usable,
                    state: frame.plan.state(),
                    restored_texels: frame.restored_texels,
                },
            ));
        }
    }

    // The views no longer cached are forgotten; everything but the map's
    // update is done before the lock.
    live.clear();
    live.extend(caches.views.keys().copied());
    reported.retain(|id| !live.contains(id));
    let Ok(mut map) = feedback.0.lock() else {
        return;
    };
    for id in reported.iter() {
        map.remove(id);
    }
    for (id, status) in statuses.drain(..) {
        map.insert(id, status);
    }
    drop(map);
    core::mem::swap(&mut *reported, &mut *live);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_light::ShadowFootprint;
    use bevy_math::Vec2;

    #[test]
    fn retiring_layers_count_until_released() {
        let mut retiring = RetiringLayers::default();
        retiring.retire(10, "a");
        retiring.tick();
        retiring.retire(5, "b");
        assert_eq!(retiring.bytes, 15);
        // "a" is held for its three frames, "b" for its own.
        retiring.tick();
        assert_eq!(retiring.bytes, 15);
        retiring.tick();
        assert_eq!(retiring.bytes, 5);
        assert_eq!(retiring.layers.len(), 1);
        retiring.tick();
        assert_eq!(retiring.bytes, 0);
        assert!(retiring.layers.is_empty());
    }

    #[test]
    fn role_keys() {
        for role in [
            ShadowViewRole::Full,
            ShadowViewRole::Static,
            ShadowViewRole::Movers,
        ] {
            for subview in [0, 5] {
                let index = role.subview_index(subview);
                assert_eq!(ShadowViewRole::of_subview_index(index), role);
                assert_eq!(index & ((1 << ShadowViewRole::SHIFT) - 1), subview);
            }
        }
        // Bevy's own keys are unchanged.
        assert_eq!(ShadowViewRole::Full.subview_index(3), 3);
        // A spot light's views keep the keys they had before views had ids;
        // a cascade's has its camera, a face its index.
        let light = Entity::from_raw_u32(4).unwrap();
        let camera = Entity::from_raw_u32(8).unwrap();
        for role in [
            ShadowViewRole::Full,
            ShadowViewRole::Static,
            ShadowViewRole::Movers,
        ] {
            assert_eq!(
                retained_shadow_view(&ShadowViewId::spot(light), role),
                RetainedViewEntity::new(MainEntity::from(light), None, role.subview_index(0))
            );
            let cascade = ShadowViewId {
                light,
                subview: bevy_light::ShadowSubview::Directional { camera, cascade: 1 },
            };
            assert_eq!(
                retained_shadow_view(&cascade, role),
                RetainedViewEntity::new(
                    MainEntity::from(light),
                    Some(MainEntity::from(camera)),
                    role.subview_index(1)
                )
            );
            let face = ShadowViewId {
                light,
                subview: bevy_light::ShadowSubview::PointFace(3),
            };
            assert_eq!(
                retained_shadow_view(&face, role),
                RetainedViewEntity::new(MainEntity::from(light), None, role.subview_index(3))
            );
        }
        assert!(ShadowViewRole::Full.takes(true) && ShadowViewRole::Full.takes(false));
        assert!(ShadowViewRole::Static.takes(true) && !ShadowViewRole::Static.takes(false));
        assert!(!ShadowViewRole::Movers.takes(true) && ShadowViewRole::Movers.takes(false));
    }

    #[test]
    fn written_tiles_over_recorded_and_unrecorded_passes() {
        let tiles = |min: Vec2, max: Vec2| {
            let mut mask = ShadowTileMask::new(1024);
            mask.insert(ShadowFootprint::Rect { min, max });
            mask
        };
        let here = tiles(Vec2::new(0.1, 0.1), Vec2::new(0.2, 0.2));
        let there = tiles(Vec2::new(0.6, 0.6), Vec2::new(0.7, 0.7));
        let mut both = here.clone();
        both.union_with(&there);
        assert!(both.texels() > here.texels());

        // A recorded pass leaves only what it drew.
        let mut written = here.clone();
        written_after_pass(&mut written, there.clone(), true);
        assert_eq!(written, there);
        // A pass that wasn't recorded may have left what was written before.
        let mut written = here.clone();
        written_after_pass(&mut written, there.clone(), false);
        assert_eq!(written, both);
        // Then a recorded pass with no movable caster: nothing is left.
        written_after_pass(&mut written, ShadowTileMask::new(1024), true);
        assert!(written.is_empty());
    }

    #[test]
    fn plans() {
        use ShadowCachePlan::*;

        // With a static layer.
        assert_eq!(ShadowCachePlan::new(true, true, true, false), Kept);
        // Movable casters now, or a trace of them to restore.
        assert_eq!(ShadowCachePlan::new(true, true, true, true), Movers);
        // Invalid static content wins over everything else.
        assert_eq!(ShadowCachePlan::new(true, false, true, true), Rebuild);
        assert_eq!(ShadowCachePlan::new(true, false, false, false), Rebuild);
        // Only the destination moved.
        assert_eq!(ShadowCachePlan::new(true, true, false, false), Restore);
        assert_eq!(ShadowCachePlan::new(true, true, false, true), Restore);

        // Without a static layer.
        assert_eq!(ShadowCachePlan::new(false, false, true, false), Kept);
        assert_eq!(ShadowCachePlan::new(false, false, false, false), Fallback);
        assert_eq!(ShadowCachePlan::new(false, false, true, true), Fallback);

        assert!(Rebuild.draws_static() && Fallback.draws_static());
        assert!(!Restore.draws_static() && !Movers.draws_static());
        assert!(Rebuild.restores_whole() && Restore.restores_whole());
        assert!(!Movers.restores_whole() && !Fallback.restores_whole() && !Kept.restores_whole());
    }

    #[test]
    fn a_moved_shadow_map_is_never_kept() {
        use ShadowCachePlan::*;
        // The shadow map's layer moved (renumbered lights, a new texture):
        // the working destination no longer matches. A light with a static
        // layer restores it whole; one without draws everything again, even
        // with no movable caster, so its map is never stale or empty.
        for movers in [false, true] {
            assert_eq!(ShadowCachePlan::new(false, false, false, movers), Fallback);
            assert_eq!(ShadowCachePlan::new(true, true, false, movers), Restore);
            assert_eq!(ShadowCachePlan::new(true, false, false, movers), Rebuild);
        }
        assert!(Fallback.draws_static());
    }
}
