//! Cached spot light shadow maps with moving casters, decided automatically.
//!
//! While [`ShadowCacheSettings::enabled`] is on, every shadowed spot light
//! keeps the shadow of its settled casters in a static layer of its own and
//! draws only its live casters every frame, on top of a copy of that layer.
//! Nobody marks casters or lights: a caster that hasn't changed settles into
//! the static layers of the lights it touches, one that changes is live again
//! in the same frame ([`classify_shadow_casters`]). This module holds what the
//! main world knows about it: each caster's state ([`ShadowCasterState`],
//! [`StaticShadowCaster`]), which settled casters each cached shadow view
//! sees and when that changes ([`ShadowViewCache`]), and which tiles of its
//! shadow map its live casters touch ([`ShadowFootprint`],
//! [`ShadowTileMask`]). The render world plans, draws and acknowledges the
//! maps (`bevy_pbr`'s `shadow_cache` module) and reports back through
//! [`ShadowCacheFeedback`].
//!
//! The cache keeps its state by shadow view ([`ShadowViewId`]): one for a
//! spot light ([`SpotLightShadowCache`]); directional cascades and point
//! light faces are reserved for later. A light keeps only what settles it
//! (what projects its maps); everything else is per view.
//!
//! # The rule
//!
//! - While the app loads ([`ShadowCacheLoading`], on until the app turns it
//!   off), a caster that appears is live in its first frame and settles in the
//!   next one if nothing about it changed: loaded content settles as fast as
//!   if it were marked static. A caster that appears later (a drop, a building
//!   piece, a short-lived spawn) waits the settle time like one that moved.
//! - A settled caster whose shadow changes (its transform, bounds, mesh, mesh
//!   tag, material, visibility or render layers, compared by value after transform
//!   propagation and bounds; or an [`InvalidateShadowCache::Caster`]) is live
//!   again in the same frame, and the lights its old footprint touched redraw
//!   their static layer.
//! - A caster that moved (its transform or bounds changed), or that was
//!   settled before, settles only after [`ShadowCacheSettings::settle_time`]
//!   of simulation time without a change. Those settlings are deferred and
//!   coalesced in waves, at most one per quarter of the settle time: a wave
//!   settles the casters that waited longest first, as long as the lights
//!   they touch stay within a budget of lights per wave (the first caster
//!   always settles), so each light redraws at most once per wave.
//! - Always live: skinned and morphed meshes, meshes with a
//!   [`VisibilityRange`], meshes whose material declares a shadow that
//!   changes by itself ([`LiveShadowMaterial`], set by `bevy_pbr` from
//!   `Material::shadow_caching`), and meshes with [`ShadowCasterLive`].
//! - Lights follow the same rule on what projects their shadow map (their
//!   transform, angle, near plane, range, render layers and the map size): a
//!   light that changed has no static layer until it settles. Its color and
//!   intensity don't count.

use alloc::sync::Arc;
use core::time::Duration;
use std::sync::Mutex;

use bevy_asset::{AssetEvent, AssetId, UntypedAssetId};
use bevy_camera::{
    primitives::{Aabb, CascadesFrusta, Frustum, Sphere},
    visibility::{InheritedVisibility, NoFrustumCulling, RenderLayers, VisibilityRange},
};
use bevy_ecs::{
    component::ComponentId,
    entity::{EntityHashMap, EntityHashSet},
    lifecycle::HookContext,
    prelude::*,
    world::DeferredWorld,
};
use bevy_math::{Affine3A, Mat4, URect, Vec2, Vec3A, Vec4Swizzles};
use bevy_mesh::{
    morph::{MeshMorphWeights, MorphWeights},
    skinning::SkinnedMesh,
    Mesh, Mesh3d, MeshTag,
};
use bevy_platform::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
};
use bevy_reflect::prelude::*;
use bevy_shape::{HalfSpace, ViewFrustum};
use bevy_time::Time;
use bevy_transform::components::GlobalTransform;

use crate::{
    spot_light_clip_from_view, spot_light_world_from_view, stable_cascade::view_takes,
    DirectionalLightShadowMap, NotShadowCaster, ShadowRepresentation, SpotLight,
};

/// The marker Bevy keeps on a mesh whose shadow is settled: it is drawn into
/// the static layers of the cached spot lights that see it, not every frame.
///
/// Bevy inserts and removes it ([`classify_shadow_casters`]); don't add it
/// yourself. Shadow systems filter meshes with it, so a cached light never
/// visits a settled mesh in a frame in which its static layer stays valid.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
#[component(on_add = on_add_static_shadow_caster, on_remove = on_remove_static_shadow_caster)]
pub struct StaticShadowCaster;

/// Keeps a mesh's shadow live: drawn every frame, never in a static layer.
///
/// For code that knows a mesh will move again soon (a lift on a schedule, a
/// door that opens every few seconds) or that draws it in a way Bevy can't
/// see (a custom render path). Without it Bevy decides by itself.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct ShadowCasterLive;

/// A spot light the cache leaves alone: drawn every frame (the audit's full
/// redraws to compare with).
#[doc(hidden)]
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ShadowCacheExempt;

/// Set by `bevy_pbr` on a mesh whose material declares a shadow that changes
/// without a component changing (vertex animation, `globals.time`, GPU-written
/// instances, animated discard): the mesh is always live, and its footprint
/// in the shadow maps is its bounds grown by `max_displacement`.
#[doc(hidden)]
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct LiveShadowMaterial {
    /// How far, in world units, the material can move a vertex out of the
    /// mesh's bounds (0: the bounds hold it).
    pub max_displacement: f32,
}

/// The material components whose material is live ([`LiveShadowMaterial`]),
/// with their displacement; `bevy_pbr` registers them.
#[doc(hidden)]
#[derive(Resource, Clone, Debug, Default)]
pub struct LiveShadowMaterials(pub Vec<(ComponentId, f32)>);

impl LiveShadowMaterials {
    /// Inserts, updates or removes an entity's [`LiveShadowMaterial`] from the
    /// material components it has now.
    pub fn sync(world: &mut World, entity: Entity) {
        let Some(materials) = world.get_resource::<LiveShadowMaterials>() else {
            return;
        };
        let Ok(entity_ref) = world.get_entity(entity) else {
            return;
        };
        let displacement = materials
            .0
            .iter()
            .filter(|(component, _)| entity_ref.contains_id(*component))
            .map(|(_, displacement)| *displacement)
            .reduce(f32::max);
        let current = entity_ref.get::<LiveShadowMaterial>().copied();
        let mut entity_mut = world.entity_mut(entity);
        match displacement {
            Some(max_displacement) => {
                let wanted = LiveShadowMaterial { max_displacement };
                if current != Some(wanted) {
                    entity_mut.insert(wanted);
                }
            }
            None if current.is_some() => {
                entity_mut.remove::<LiveShadowMaterial>();
            }
            None => {}
        }
    }
}

/// Settings of the spot light shadow cache.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[reflect(Resource, Default, Debug, Clone, PartialEq)]
pub struct ShadowCacheSettings {
    /// Caches every shadowed spot light. Off (Bevy's behavior), every shadow
    /// map is drawn every frame.
    pub enabled: bool,
    /// How many bytes the static layers of all cached spot lights may use
    /// together. A 4096² layer takes 64 MiB, a 2048² layer 16 MiB. A light
    /// that doesn't fit is drawn whole every frame in which something in its
    /// frustum is live: the same shadow map at the cost of time. Changing it
    /// redraws every cached light.
    pub budget_bytes: u64,
    /// How long, in simulation time ([`Time`]), a caster that moved or was
    /// settled before, or a light that changed, must stay unchanged before it
    /// settles. Changing it redraws nothing.
    pub settle_time: Duration,
}

impl Default for ShadowCacheSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            budget_bytes: 256 * 1024 * 1024,
            settle_time: Duration::from_secs(1),
        }
    }
}

/// Whether the app is loading its content: while it is (the default), a
/// caster that appears settles in the next frame if unchanged; once the app
/// sets it to `false` (its load gate passed), a caster that appears waits
/// [`ShadowCacheSettings::settle_time`], so short-lived spawns never touch a
/// static layer. Flipping it redraws nothing.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Reflect)]
#[reflect(Resource, Default, Debug, Clone, PartialEq)]
pub struct ShadowCacheLoading(pub bool);

impl Default for ShadowCacheLoading {
    fn default() -> Self {
        Self(true)
    }
}

/// Faults to inject into the spot light shadow cache, for tests that must see
/// a broken cache fail (the `spotlight` example's oracle). All off by default.
#[doc(hidden)]
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowCacheFaults {
    /// Never restore the static layer into the shadow map.
    pub skip_restore: bool,
    /// Restore only where live casters are now, not where they were drawn.
    pub drop_written: bool,
    /// Leave one tile out of a movers restore: one where live casters were
    /// drawn and are no longer, if there is one.
    pub drop_one_tile: bool,
    /// Don't redraw the lights a settled caster that changed (or left) was
    /// in.
    pub miss_static_invalidation: bool,
    /// Don't redraw the lights a caster that settled is in.
    pub miss_settle_invalidation: bool,
    /// Don't see a change of a caster's [`MeshTag`].
    pub miss_tag_change: bool,
    /// Insert and remove [`StaticShadowCaster`] a frame after the lights were
    /// redrawn for it.
    pub late_marker: bool,
    /// The main world takes the acknowledgement of the previous static
    /// revision for the current one.
    pub stale_ack: bool,
    /// Restore into the next layer of the shadow map texture.
    pub wrong_layer: bool,
    /// Acknowledge a static layer without checking that its casters were
    /// ready.
    pub acknowledge_early: bool,
    /// In the frame a light lights a view again, restore only the tiles live
    /// casters touch now, not those they were drawn into before it lit no
    /// view.
    pub skip_reentry_restore: bool,
    /// Take a cached directional cascade's map for current although another
    /// camera drew its own cascade into the same layer since.
    pub miss_ownership: bool,
    /// Keep a cached directional cascade's view and static layer when its
    /// projection changes (a re-center, a new depth interval).
    pub stale_projection: bool,
    /// Find where a cached far cascade's live casters fall with the cascade
    /// before it.
    pub wrong_cascade: bool,
    /// Never re-center a stable far cascade (a projection fault: its coverage
    /// expires; caught by the coverage check, not by the maps).
    pub miss_recenter: bool,
    /// Never re-center the middle stable cascade (the one before the last).
    pub miss_recenter_middle: bool,
    /// Let every cached directional view of a light that settled enter in
    /// the same frame (all of them rebuild at once), not one a frame.
    pub no_stagger: bool,
    /// Cache a directional view in the frame its projection changed, without
    /// waiting for it to settle.
    pub early_view_settle: bool,
    /// The middle stable cascade draws the far cascade's stand-ins instead
    /// of its own.
    pub middle_stand_in_routing: bool,
    /// A stable cascade that fell back to Bevy's projection never returns.
    pub stuck_stock: bool,
    /// A stable cascade never falls back to Bevy's projection, however
    /// often it re-centers.
    pub no_speed_fallback: bool,
    /// Not a fault: each cached directional cascade also lists every caster
    /// it draws, found anew each frame, for the audit's reference view
    /// (`bevy_pbr`'s `ShadowCacheAuditPlugin` sets it).
    pub directional_reference: bool,
    /// Not a fault: treat the blit that restores static layers as not
    /// compiled, so every cached light falls back to full redraws, as in the
    /// first frames. The shadow maps must stay correct.
    pub restore_not_ready: bool,
    /// Not a fault: a mesh's shadow pipeline counts as not ready in every
    /// shadow view for this many frames after the mesh first appears in one,
    /// as when its shader compiles late. The shadow maps must stay correct.
    pub readiness_delay_frames: u32,
}

/// Reports a change Bevy can't see.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidateShadowCache {
    /// A caster's shadow changed (a material that starts to discard, a vertex
    /// shader that moved its vertices): it is live again, as if it moved.
    Caster(Entity),
    /// This light redraws its static layer.
    Light(Entity),
    /// Every cached light redraws its static layer.
    All,
}

/// One shadow map a light draws: the logical identity the cache keeps its
/// state by, whatever texture, layer or rectangle the map is drawn into.
///
/// A spot light draws one view; a directional light one per camera and
/// cascade; a point light one per cube face. The entities are the main
/// world's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Reflect)]
#[reflect(Debug, Clone, PartialEq, Hash)]
pub struct ShadowViewId {
    /// The light.
    pub light: Entity,
    /// Which of the light's shadow maps.
    pub subview: ShadowSubview,
}

/// Which of a light's shadow maps a [`ShadowViewId`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Reflect)]
#[reflect(Debug, Clone, PartialEq, Hash)]
pub enum ShadowSubview {
    /// A spot light's one shadow map.
    Spot,
    /// A cascade a directional light draws for a camera.
    Directional {
        /// The camera.
        camera: Entity,
        /// The cascade's index.
        cascade: u32,
    },
    /// A face of a point light's cube map.
    PointFace(u8),
}

impl ShadowViewId {
    /// The shadow view of a spot light.
    pub fn spot(light: Entity) -> ShadowViewId {
        ShadowViewId {
            light,
            subview: ShadowSubview::Spot,
        }
    }

    /// The camera the view follows (a directional light's cascades).
    pub fn camera(&self) -> Option<Entity> {
        match self.subview {
            ShadowSubview::Directional { camera, .. } => Some(camera),
            ShadowSubview::Spot | ShadowSubview::PointFace(_) => None,
        }
    }

    /// The face or cascade index, as Bevy numbers the light's shadow views
    /// (a spot light's is 0).
    pub fn subview_index(&self) -> u32 {
        match self.subview {
            ShadowSubview::Spot => 0,
            ShadowSubview::Directional { cascade, .. } => cascade,
            ShadowSubview::PointFace(face) => face as u32,
        }
    }
}

/// The generation of a shadow view's cached state: a new one each time the
/// main world creates the state (a light that gets its cache component again
/// starts over), so that the render world's reports on an earlier state can't
/// validate this one. 0 until the state is first used.
fn next_shadow_view_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// What last changed about a caster's shadow, as [`ShadowCasterState`]
/// reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
#[reflect(Default, Debug, Clone, PartialEq)]
pub enum ShadowCasterChange {
    /// Nothing yet: it appeared.
    #[default]
    New,
    /// Its transform.
    Transform,
    /// Its bounds ([`Aabb`] or [`NoFrustumCulling`]).
    Bounds,
    /// Its mesh, its mesh asset, or its [`MeshTag`] (which a shader may read
    /// to place its vertices).
    Mesh,
    /// Its material, or its material asset.
    Material,
    /// Its visibility.
    Visibility,
    /// Its render layers.
    Layers,
    /// An [`InvalidateShadowCache::Caster`].
    Reported,
    /// Something that kept it live went (a skinned mesh, morph weights, a
    /// [`VisibilityRange`], a live material, [`ShadowCasterLive`]), or it
    /// stopped being a caster.
    Forced,
}

impl ShadowCasterChange {
    /// Whether the change is a move: a caster that moved settles only after
    /// [`ShadowCacheSettings::settle_time`].
    fn is_motion(self) -> bool {
        matches!(
            self,
            ShadowCasterChange::Transform
                | ShadowCasterChange::Bounds
                | ShadowCasterChange::Reported
                | ShadowCasterChange::Forced
        )
    }
}

/// What decides a caster's shadow, as last seen. Compared by value: Bevy
/// marks writes, not changes.
#[derive(Clone, Debug)]
pub(crate) struct CasterSignature {
    world_from_local: Affine3A,
    aabb: Option<Aabb>,
    mesh: AssetId<Mesh>,
    /// Its [`MeshTag`]'s value (0 without one, as the shaders see it).
    tag: u32,
    visible: bool,
    layers: RenderLayers,
    no_frustum_culling: bool,
    /// MPS: which shadow views it casts into besides its layers.
    representation: Option<ShadowRepresentation>,
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

impl CasterSignature {
    /// What differs from `other`, bit for bit, if anything.
    fn change_from(&self, other: &CasterSignature) -> Option<ShadowCasterChange> {
        let affine = |signature: &CasterSignature| signature.world_from_local.to_cols_array();
        let aabb = |aabb: &Aabb| {
            let (center, half_extents) = (aabb.center, aabb.half_extents);
            [
                center.x,
                center.y,
                center.z,
                half_extents.x,
                half_extents.y,
                half_extents.z,
            ]
        };
        if !same_bits(&affine(self), &affine(other)) {
            Some(ShadowCasterChange::Transform)
        } else if self.no_frustum_culling != other.no_frustum_culling
            || match (&self.aabb, &other.aabb) {
                (Some(a), Some(b)) => !same_bits(&aabb(a), &aabb(b)),
                (None, None) => false,
                _ => true,
            }
        {
            Some(ShadowCasterChange::Bounds)
        } else if self.mesh != other.mesh || self.tag != other.tag {
            Some(ShadowCasterChange::Mesh)
        } else if self.visible != other.visible {
            Some(ShadowCasterChange::Visibility)
        } else if self.layers != other.layers || self.representation != other.representation {
            Some(ShadowCasterChange::Layers)
        } else {
            None
        }
    }

    /// Whether a caster with this signature can be drawn by a shadow view.
    fn touches(&self, light: &ShadowViewBounds) -> bool {
        self.visible
            && light.layers.intersects(&self.layers)
            && view_takes(light.stand_ins, self.representation)
            && match &self.aabb {
                Some(aabb) if !self.no_frustum_culling => {
                    light
                        .sphere
                        .is_none_or(|sphere| sphere.intersects_obb(aabb, &self.world_from_local))
                        && light
                            .frustum
                            .intersects_obb(aabb, &self.world_from_local, true, true)
                }
                _ => true,
            }
    }
}

/// Where a cached shadow view can draw a caster.
pub(crate) struct ShadowViewBounds {
    /// Its range (a directional view has none).
    sphere: Option<Sphere>,
    frustum: Frustum,
    layers: RenderLayers,
    /// The stand-ins it draws instead of the detail.
    stand_ins: Option<u32>,
}

impl ShadowViewBounds {
    /// A spot light's view: its range, its frustum and the layers that
    /// projected its static layer.
    fn spot(
        spot_light: &SpotLight,
        transform: &GlobalTransform,
        frustum: &Frustum,
        layers: RenderLayers,
    ) -> ShadowViewBounds {
        ShadowViewBounds {
            sphere: Some(Sphere {
                center: transform.translation_vec3a(),
                radius: spot_light.range,
            }),
            frustum: *frustum,
            layers,
            stand_ins: None,
        }
    }

    /// A directional light's cascade: its frustum without its near plane (a
    /// caster before it is drawn, clamped) and the layers that projected its
    /// static layer.
    pub(crate) fn directional(
        frustum: &Frustum,
        layers: RenderLayers,
        stand_ins: Option<u32>,
    ) -> ShadowViewBounds {
        let mut frustum = *frustum;
        frustum.half_spaces[ViewFrustum::NEAR_PLANE_IDX] = HalfSpace::new(
            frustum.half_spaces[ViewFrustum::NEAR_PLANE_IDX]
                .normal()
                .extend(f32::INFINITY),
        );
        ShadowViewBounds {
            sphere: None,
            frustum,
            layers,
            stand_ins,
        }
    }
}

/// A caster's state in the shadow cache, kept by Bevy on every [`Mesh3d`]
/// ([`classify_shadow_casters`]). Read it to see why a caster is live; it is
/// settled while it has [`StaticShadowCaster`].
#[derive(Component, Clone, Debug, Default)]
pub struct ShadowCasterState {
    /// As last seen; while settled, as when it settled.
    signature: Option<CasterSignature>,
    /// Its material, as `bevy_pbr` last saw it.
    material: Option<UntypedAssetId>,
    /// A change `bevy_pbr` found.
    pending: Option<ShadowCasterChange>,
    /// It moved, or was settled before: it settles only after the settle
    /// time.
    moved: bool,
    /// The classification frame it was last seen in while live.
    observed: u32,
    /// The classification frame of its last change (or first sight).
    changed_frame: u32,
    /// The simulation time of its last change.
    last_change: Duration,
    reason: ShadowCasterChange,
}

impl ShadowCasterState {
    /// What last changed about its shadow.
    pub fn reason(&self) -> ShadowCasterChange {
        self.reason
    }

    /// The simulation time of the last change.
    pub fn last_change(&self) -> Duration {
        self.last_change
    }

    /// Whether it moved, or was settled before: it then settles only after
    /// [`ShadowCacheSettings::settle_time`].
    pub fn moved(&self) -> bool {
        self.moved
    }

    /// For `bevy_pbr`: the caster's material as last noted.
    #[doc(hidden)]
    pub fn material(&self) -> Option<UntypedAssetId> {
        self.material
    }

    /// For `bevy_pbr`: notes the caster's material; a change makes the
    /// caster live (it is classified this frame if this runs before
    /// [`classify_shadow_casters`]).
    #[doc(hidden)]
    pub fn note_material(&mut self, material: UntypedAssetId) {
        if self
            .material
            .replace(material)
            .is_some_and(|old| old != material)
        {
            self.pending = Some(ShadowCasterChange::Material);
        }
    }

    /// For `bevy_pbr`: the caster's material asset changed.
    #[doc(hidden)]
    pub fn note_material_modified(&mut self) {
        self.pending = Some(ShadowCasterChange::Material);
    }

    fn change(&mut self, reason: ShadowCasterChange, frame: u32, now: Duration) {
        self.reason = reason;
        self.changed_frame = frame;
        self.last_change = now;
        if reason.is_motion() {
            self.moved = true;
        }
    }
}

/// The footprints of casters that settled or left the static layers since
/// [`invalidate_shadow_caches`] last ran, from [`StaticShadowCaster`]'s
/// hooks: the lights they touch redraw. `None`: unknown, every light.
#[derive(Resource, Default)]
#[doc(hidden)]
pub struct ShadowCasterTransitions {
    settled: Vec<Option<CasterSignature>>,
    left: Vec<Option<CasterSignature>>,
}

fn on_add_static_shadow_caster(mut world: DeferredWorld, context: HookContext) {
    let signature = world
        .get::<ShadowCasterState>(context.entity)
        .and_then(|state| state.signature.clone());
    if let Some(mut transitions) = world.get_resource_mut::<ShadowCasterTransitions>() {
        transitions.settled.push(signature);
    }
}

fn on_remove_static_shadow_caster(mut world: DeferredWorld, context: HookContext) {
    // The signature as it settled: where its shadow is in the static layers.
    let signature = world
        .get::<ShadowCasterState>(context.entity)
        .and_then(|state| state.signature.clone());
    if let Some(mut transitions) = world.get_resource_mut::<ShadowCasterTransitions>() {
        transitions.left.push(signature);
    }
}

/// Counts of the cache's decisions, for tools (`stat` overlays) and tests:
/// a caster that goes live without a reason costs rebuilds while the image
/// stays right, so only these show it.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[reflect(Resource, Default, Debug, Clone, PartialEq)]
pub struct ShadowCacheCounters {
    /// Settled casters that went live again (promotions), since the start.
    pub promotions: u64,
    /// Casters that settled after the settle time (demotions), since the
    /// start.
    pub demotions: u64,
    /// Casters that settled in the frame after they appeared, since the start.
    pub first_settles: u64,
    /// Requests to redraw a cached light's static layer, since the start.
    pub rebuild_requests: u64,
    /// Casters live now that can settle (not forced live).
    pub live: u32,
    /// Of them, those waiting for the settle time.
    pub settling: u32,
    /// Of them, those past it and waiting for the next wave.
    pub deferred: u32,
    /// Of them, those that appeared while loading (or changed without moving
    /// before they first settled) and settle in the next frame: 0 once loaded
    /// content has settled.
    pub fresh: u32,
    /// Casters forced live now (skinned, morphed, `VisibilityRange`, a live
    /// material, [`ShadowCasterLive`]).
    pub forced_live: u32,
    /// Settled casters now.
    pub settled: u32,
}

/// Where the movable casters of a cached spot light fall in its shadow map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[reflect(Default, Debug, Clone, PartialEq)]
pub enum ShadowFootprint {
    /// Nowhere.
    #[default]
    Empty,
    /// Inside this rectangle of the shadow map's uv space (`[0, 1]²`, v down,
    /// as texels are addressed).
    Rect {
        /// The smallest corner.
        min: Vec2,
        /// The largest corner.
        max: Vec2,
    },
    /// Anywhere: a caster crosses the light's near plane or its bounds aren't
    /// known.
    Whole,
}

impl ShadowFootprint {
    /// Whether no texel is covered.
    pub fn is_empty(&self) -> bool {
        matches!(self, ShadowFootprint::Empty)
    }

    /// The footprint of an oriented box (`aabb` placed by `world_from_local`)
    /// in a perspective shadow map, `clip_from_world` being Bevy's reversed
    /// infinite projection with this `near_z`.
    ///
    /// A box with a corner closer to the light than the near plane covers the
    /// whole map: its clipped triangles can reach any texel.
    pub fn of_box(
        clip_from_world: &Mat4,
        near_z: f32,
        aabb: &Aabb,
        world_from_local: &Affine3A,
    ) -> ShadowFootprint {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        // Clip space is linear in the corner: the center's image plus or
        // minus the images of the half extents along each local axis.
        let clip_from_local = *clip_from_world * Mat4::from(*world_from_local);
        let center = clip_from_local * aabb.center.extend(1.0);
        let axes = [
            clip_from_local.x_axis * aabb.half_extents.x,
            clip_from_local.y_axis * aabb.half_extents.y,
            clip_from_local.z_axis * aabb.half_extents.z,
        ];
        for corner in 0..8 {
            let mut clip = center;
            for (bit, axis) in axes.iter().enumerate() {
                if corner & (1 << bit) == 0 {
                    clip -= *axis;
                } else {
                    clip += *axis;
                }
            }
            // In Bevy's perspective projections `w` is the distance in front of
            // the light along its axis.
            if clip.w.is_nan() || clip.w < near_z {
                return ShadowFootprint::Whole;
            }
            let ndc = clip.xy() / clip.w;
            let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            min = min.min(uv);
            max = max.max(uv);
        }
        let min = min.max(Vec2::ZERO);
        let max = max.min(Vec2::ONE);
        if min.x > max.x || min.y > max.y {
            return ShadowFootprint::Empty;
        }
        ShadowFootprint::Rect { min, max }
    }

    /// The footprint of an oriented box in an orthographic shadow map (a
    /// directional light's cascade). Its depth doesn't matter: a caster
    /// closer to the light than the near plane is drawn clamped onto it,
    /// where its box projects.
    pub fn of_box_orthographic(
        clip_from_world: &Mat4,
        aabb: &Aabb,
        world_from_local: &Affine3A,
    ) -> ShadowFootprint {
        let clip_from_local = *clip_from_world * Mat4::from(*world_from_local);
        let center = clip_from_local * aabb.center.extend(1.0);
        let reach = (clip_from_local.x_axis * aabb.half_extents.x).abs()
            + (clip_from_local.y_axis * aabb.half_extents.y).abs()
            + (clip_from_local.z_axis * aabb.half_extents.z).abs();
        if !(center.is_finite() && reach.is_finite()) {
            return ShadowFootprint::Whole;
        }
        let (low, high) = (center.xy() - reach.xy(), center.xy() + reach.xy());
        // v runs down the map.
        let min = Vec2::new(low.x * 0.5 + 0.5, 0.5 - high.y * 0.5).max(Vec2::ZERO);
        let max = Vec2::new(high.x * 0.5 + 0.5, 0.5 - low.y * 0.5).min(Vec2::ONE);
        if min.x > max.x || min.y > max.y {
            return ShadowFootprint::Empty;
        }
        ShadowFootprint::Rect { min, max }
    }
}

/// The tiles of a cached spot light's shadow map that movable casters touch:
/// a bitset of square tiles of [`ShadowTileMask::TILE`] texels, row by row.
///
/// A tile is set when a caster's footprint, with a margin of one texel for
/// rounding, reaches into it; a restore covers the set tiles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShadowTileMask {
    /// The shadow map's size in texels.
    size: u32,
    /// Tiles per side.
    side: u32,
    /// 64-bit words per row of tiles.
    words_per_row: u32,
    bits: Vec<u64>,
}

impl ShadowTileMask {
    /// The side of a tile in texels.
    pub const TILE: u32 = 16;

    /// An empty mask for a shadow map of `size` texels.
    pub fn new(size: u32) -> ShadowTileMask {
        let mut mask = ShadowTileMask::default();
        mask.reset(size);
        mask
    }

    /// Clears the mask, for a shadow map of `size` texels.
    pub fn reset(&mut self, size: u32) {
        self.size = size;
        self.side = size.div_ceil(Self::TILE);
        self.words_per_row = self.side.div_ceil(64);
        self.bits.clear();
        self.bits
            .resize((self.side * self.words_per_row) as usize, 0);
    }

    /// The shadow map's size in texels.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Whether no tile is set.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|word| *word == 0)
    }

    /// The share of the map's tiles that are set, from 0 to 1.
    pub fn coverage(&self) -> f32 {
        let set: u32 = self.bits.iter().map(|word| word.count_ones()).sum();
        set as f32 / (self.side * self.side).max(1) as f32
    }

    /// Sets every tile.
    pub fn fill(&mut self) {
        if self.side > 0 {
            self.set_tiles(0..self.side, 0, self.side - 1);
        }
    }

    /// Sets the tiles a [`ShadowFootprint`] can touch.
    pub fn insert(&mut self, footprint: ShadowFootprint) {
        let ShadowFootprint::Rect { min, max } = footprint else {
            if footprint == ShadowFootprint::Whole {
                self.fill();
            }
            return;
        };
        // The texels it can touch, with a margin of one texel for rounding.
        let size = Vec2::splat(self.size as f32);
        let min = ((min * size).floor() - 1.0).clamp(Vec2::ZERO, size);
        let max = ((max * size).ceil() + 1.0).clamp(Vec2::ZERO, size);
        if min.x >= max.x || min.y >= max.y {
            return;
        }
        let tile = Self::TILE;
        let (x0, y0) = (min.x as u32 / tile, min.y as u32 / tile);
        let (x1, y1) = ((max.x as u32 - 1) / tile, (max.y as u32 - 1) / tile);
        self.set_tiles(y0..y1 + 1, x0, x1);
    }

    /// Sets the tiles `first..=last` of these rows.
    fn set_tiles(&mut self, rows: core::ops::Range<u32>, first: u32, last: u32) {
        for row in rows {
            let row_start = (row * self.words_per_row) as usize;
            for word in first / 64..=last / 64 {
                let low = first.max(word * 64) - word * 64;
                let high = last.min(word * 64 + 63) - word * 64;
                self.bits[row_start + word as usize] |= (!0u64 >> (63 - high)) & (!0u64 << low);
            }
        }
    }

    /// Sets the tiles set in `other`, or every tile if `other` is for another
    /// map size and isn't empty.
    pub fn union_with(&mut self, other: &ShadowTileMask) {
        if other.size != self.size {
            if !other.is_empty() {
                self.fill();
            }
            return;
        }
        for (word, other) in self.bits.iter_mut().zip(&other.bits) {
            *word |= *other;
        }
    }

    /// Calls `f` with the texels of each run of set tiles in a row, row by
    /// row, clamped to the map.
    pub fn for_each_run(&self, mut f: impl FnMut(URect)) {
        let tile = Self::TILE;
        let size = self.size;
        for row in 0..self.side {
            let mut emit = |(start, end): (u32, u32)| {
                f(URect::new(
                    start * tile,
                    row * tile,
                    (end * tile).min(size),
                    ((row + 1) * tile).min(size),
                ));
            };
            let mut run: Option<(u32, u32)> = None;
            let row_start = (row * self.words_per_row) as usize;
            for word_index in 0..self.words_per_row {
                let mut word = self.bits[row_start + word_index as usize];
                while word != 0 {
                    let start = word.trailing_zeros();
                    let len = (word >> start).trailing_ones();
                    let tiles = (word_index * 64 + start, word_index * 64 + start + len);
                    run = match run {
                        Some((run_start, run_end)) if run_end == tiles.0 => {
                            Some((run_start, tiles.1))
                        }
                        Some(previous) => {
                            emit(previous);
                            Some(tiles)
                        }
                        None => Some(tiles),
                    };
                    word = if start + len >= 64 {
                        0
                    } else {
                        word & (!0u64 << (start + len))
                    };
                }
            }
            if let Some(run) = run {
                emit(run);
            }
        }
    }

    /// How many texels the set tiles cover.
    pub fn texels(&self) -> u64 {
        let mut texels = 0;
        self.for_each_run(|rect| texels += rect.width() as u64 * rect.height() as u64);
        texels
    }

    /// For the `drop_one_tile` fault: clears one set tile, the first one not
    /// set in `now` if there is one.
    #[doc(hidden)]
    pub fn drop_one_tile(&mut self, now: &ShadowTileMask) {
        let now_word = |index: usize| now.bits.get(index).copied().unwrap_or(0);
        let index = (0..self.bits.len())
            .find(|&index| self.bits[index] & !now_word(index) != 0)
            .map(|index| (index, self.bits[index] & !now_word(index)))
            .or_else(|| {
                (0..self.bits.len())
                    .find(|&index| self.bits[index] != 0)
                    .map(|index| (index, self.bits[index]))
            });
        if let Some((index, word)) = index {
            self.bits[index] &= !(1u64 << word.trailing_zeros());
        }
    }
}

/// What decides a cached spot light's static layer besides its casters: when
/// any of it changes, the static layer is drawn again.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpotShadowProjection {
    world_from_view: Affine3A,
    outer_angle: f32,
    near_z: f32,
    range: f32,
    map_size: usize,
    render_layers: RenderLayers,
}

/// The clamped outer angle that the render world projects a spot light's
/// shadow map with.
pub(crate) fn spot_shadow_outer_angle(spot_light: &SpotLight) -> f32 {
    spot_light
        .outer_angle
        .min(core::f32::consts::FRAC_PI_2 - 1e-4)
}

impl SpotShadowProjection {
    pub(crate) fn new(
        transform: &GlobalTransform,
        spot_light: &SpotLight,
        render_layers: &RenderLayers,
        map_size: usize,
    ) -> Self {
        Self {
            world_from_view: spot_light_world_from_view(transform),
            outer_angle: spot_shadow_outer_angle(spot_light),
            near_z: spot_light.shadow_map_near_z,
            range: spot_light.range,
            map_size,
            render_layers: render_layers.clone(),
        }
    }

    /// The light's clip-from-world matrix, as its shadow view renders.
    pub(crate) fn clip_from_world(&self) -> Mat4 {
        spot_light_clip_from_view(self.outer_angle, self.near_z)
            * Mat4::from(self.world_from_view.inverse())
    }

    pub(crate) fn near_z(&self) -> f32 {
        self.near_z
    }
}

/// The movable shadow casters of a frame, bucketed by where they are, so that
/// a cached spot light tests only the casters near it instead of every movable
/// caster in the world.
///
/// The buckets are columns of [`Self::CELL`] x [`Self::CELL`] on the
/// horizontal plane (X and Z; Y is up), sorted by column. A caster goes into
/// the column of its center; one wider than half a column, or without bounds,
/// is a candidate of every light.
#[derive(Default)]
#[doc(hidden)]
pub struct MoversGrid {
    /// By column (z, x), sorted.
    bucketed: Vec<((i32, i32), Entity)>,
    /// Casters every light tests.
    everywhere: Vec<Entity>,
}

impl MoversGrid {
    /// The width of a column, in world units (meters by convention): about
    /// a character's reach, so a column holds few casters.
    pub(crate) const CELL: f32 = 4.0;

    fn column(x: f32, z: f32) -> (i32, i32) {
        (
            (z / Self::CELL).floor() as i32,
            (x / Self::CELL).floor() as i32,
        )
    }

    /// The world-space center and half extents of the box that holds a
    /// caster's oriented bounds.
    pub(crate) fn world_bounds(aabb: &Aabb, transform: &GlobalTransform) -> (Vec3A, Vec3A) {
        let world_from_local = transform.affine();
        let axes = world_from_local.matrix3;
        let half_extents = axes.x_axis.abs() * aabb.half_extents.x
            + axes.y_axis.abs() * aabb.half_extents.y
            + axes.z_axis.abs() * aabb.half_extents.z;
        (
            world_from_local.transform_point3a(aabb.center),
            half_extents,
        )
    }

    /// Fills the grid from this frame's visible movable casters: `(entity,
    /// world-space bounds)`, the bounds `None` for a caster without them
    /// (see [`Self::world_bounds`]).
    pub(crate) fn build(
        &mut self,
        casters: impl Iterator<Item = (Entity, Option<(Vec3A, Vec3A)>)>,
    ) {
        self.bucketed.clear();
        self.everywhere.clear();
        for (entity, bounds) in casters {
            match bounds {
                Some((center, half_extents))
                    if center.is_finite()
                        && half_extents.x <= Self::CELL * 0.5
                        && half_extents.z <= Self::CELL * 0.5 =>
                {
                    self.bucketed
                        .push((Self::column(center.x, center.z), entity));
                }
                _ => self.everywhere.push(entity),
            }
        }
        self.bucketed.sort_unstable_by_key(|(column, _)| *column);
    }

    /// How many casters it holds.
    pub(crate) fn len(&self) -> usize {
        self.bucketed.len() + self.everywhere.len()
    }

    /// Every caster that may intersect a sphere (a light's range), and some
    /// that don't, appended to `candidates`.
    pub(crate) fn candidates(
        &self,
        center: Vec3A,
        radius: f32,
        candidates: &mut impl Extend<Entity>,
    ) {
        candidates.extend(self.everywhere.iter().copied());
        if !(center.is_finite() && radius.is_finite()) {
            candidates.extend(self.bucketed.iter().map(|(_, entity)| *entity));
            return;
        }
        // A caster's center is at most half a column from its bounds.
        let reach = radius + Self::CELL * 0.5;
        let (z_min, x_min) = Self::column(center.x - reach, center.z - reach);
        let (z_max, x_max) = Self::column(center.x + reach, center.z + reach);
        if (z_max as i64 - z_min as i64) > self.bucketed.len() as i64 {
            // A light larger than the level: every caster.
            candidates.extend(self.bucketed.iter().map(|(_, entity)| *entity));
            return;
        }
        for z in z_min..=z_max {
            let start = self
                .bucketed
                .partition_point(|(column, _)| *column < (z, x_min));
            let end = self
                .bucketed
                .partition_point(|(column, _)| *column <= (z, x_max));
            candidates.extend(self.bucketed[start..end].iter().map(|(_, entity)| *entity));
        }
    }
}

/// What decides a cached directional cascade's static layer besides its
/// casters: its exact projection (a stable far cascade keeps it bit for bit
/// while the camera only turns), the map size, the layers and the stand-ins
/// it draws instead of the detail.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectionalShadowProjection {
    clip_from_world: [u32; 16],
    map_size: usize,
    render_layers: RenderLayers,
    stand_ins: Option<u32>,
}

impl DirectionalShadowProjection {
    pub(crate) fn new(
        clip_from_world: &Mat4,
        map_size: usize,
        render_layers: &RenderLayers,
        stand_ins: Option<u32>,
    ) -> Self {
        Self {
            clip_from_world: clip_from_world.to_cols_array().map(f32::to_bits),
            map_size,
            render_layers: render_layers.clone(),
            stand_ins,
        }
    }

    /// The stand-ins the view draws instead of the detail.
    pub(crate) fn stand_ins(&self) -> Option<u32> {
        self.stand_ins
    }
}

/// What a directional light's cached views settle on: what projects every
/// cascade of every camera (its direction, the map size and its layers). A
/// camera's far projection moving (a re-center) redraws that view but doesn't
/// restart the light's settling.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectionalLightSettle {
    pub(crate) rotation: [u32; 4],
    pub(crate) map_size: usize,
    pub(crate) render_layers: RenderLayers,
}

/// What projected a shadow view's static layer: when it changes, the layer
/// is drawn again. Each light type's adapter fills in its own.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShadowViewProjection {
    /// A spot light's.
    Spot(SpotShadowProjection),
    /// A directional light's cascade for a camera.
    Directional(DirectionalShadowProjection),
}

impl ShadowViewProjection {
    /// The render layers the view draws.
    pub(crate) fn render_layers(&self) -> &RenderLayers {
        match self {
            ShadowViewProjection::Spot(projection) => &projection.render_layers,
            ShadowViewProjection::Directional(projection) => &projection.render_layers,
        }
    }

    /// Whether it is this spot light projection.
    pub(crate) fn is_spot(&self, spot: &SpotShadowProjection) -> bool {
        match self {
            ShadowViewProjection::Spot(projection) => projection == spot,
            ShadowViewProjection::Directional(_) => false,
        }
    }

    /// Whether it is this directional cascade projection.
    pub(crate) fn is_directional(&self, directional: &DirectionalShadowProjection) -> bool {
        match self {
            ShadowViewProjection::Directional(projection) => projection == directional,
            ShadowViewProjection::Spot(_) => false,
        }
    }
}

/// What the main world keeps for one cached shadow view ([`ShadowViewId`]):
/// the settled casters in its frustum, the revision of that set, the
/// projection that drew its static layer, and where its live casters are.
#[derive(Clone, Debug, Default)]
pub struct ShadowViewCache {
    /// See [`ShadowViewCache::generation`].
    pub(crate) generation: u64,
    pub(crate) static_entities: Vec<Entity>,
    /// While it has no usable static layer: a print of its settled casters
    /// instead of [`Self::static_entities`].
    pub(crate) static_print: Option<crate::StaticCastersPrint>,
    pub(crate) static_revision: u64,
    pub(crate) statics_walked: bool,
    pub(crate) movers_tiles: ShadowTileMask,
    /// What projected the static layer of [`Self::static_revision`].
    pub(crate) projection: Option<ShadowViewProjection>,
    pub(crate) invalidated: bool,
    pub(crate) live_casters: u32,
    pub(crate) lights_a_view: bool,
    pub(crate) single_view: bool,
    /// For the audit: every caster the view draws, found anew this frame
    /// (directional cascades, with
    /// [`ShadowCacheFaults::directional_reference`]).
    pub(crate) reference: Option<Vec<Entity>>,
}

impl ShadowViewCache {
    /// Changes when the main world creates the view's state anew (a light
    /// that gets its cache component again starts over): the render world's
    /// state and reports for another generation are not this view's. 0 until
    /// the view is first used.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Gives a new view its generation.
    pub(crate) fn begin(&mut self) {
        if self.generation == 0 {
            self.generation = next_shadow_view_generation();
        }
    }

    /// The render world's report on this view, if it is about this
    /// generation.
    pub(crate) fn status<'a>(
        &self,
        statuses: &'a HashMap<ShadowViewId, ShadowCacheViewStatus>,
        id: &ShadowViewId,
    ) -> Option<&'a ShadowCacheViewStatus> {
        statuses
            .get(id)
            .filter(|status| status.generation == self.generation)
    }

    /// The settled casters in the view's frustum, sorted, as of
    /// [`Self::static_revision`].
    pub fn static_entities(&self) -> &[Entity] {
        &self.static_entities
    }

    /// Changes whenever the view's static layer must be drawn again.
    pub fn static_revision(&self) -> u64 {
        self.static_revision
    }

    /// Whether [`Self::static_entities`] were found, and marked visible, in
    /// this frame. The render world draws the static layer only from such a
    /// frame's list.
    pub fn statics_walked(&self) -> bool {
        self.statics_walked
    }

    /// The tiles of the view's shadow map its live casters touch in this
    /// frame.
    pub fn movers_tiles(&self) -> &ShadowTileMask {
        &self.movers_tiles
    }

    /// The live casters in the view's frustum this frame (0 while its light
    /// isn't cached, or lights no view).
    pub fn live_casters(&self) -> u32 {
        self.live_casters
    }

    /// Whether the view's light reaches into a camera's view this frame. A
    /// view whose light lights none finds neither its live nor its settled
    /// casters, and the render world draws nothing for it: its shadow map
    /// keeps the live casters it was last drawn with, and the frame its light
    /// lights a view again restores their tiles (or rebuilds) as after any
    /// frame.
    pub fn lights_a_view(&self) -> bool {
        self.lights_a_view
    }

    /// For the audit: every caster the view draws, found anew this frame,
    /// when [`ShadowCacheFaults::directional_reference`] asks for it.
    #[doc(hidden)]
    pub fn reference_entities(&self) -> Option<&[Entity]> {
        self.reference.as_deref()
    }

    /// Whether the view has no static layer it can use (the render world
    /// reported a fallback): its live and settled casters are then found as
    /// for an uncached light, in one list drawn by one view, its
    /// [`Self::movers_tiles`] all set while any live caster is in its
    /// frustum.
    pub fn single_view(&self) -> bool {
        self.single_view
    }
}

/// How a light settles: what projects its shadow maps, as last seen, and
/// when it last changed. A light whose projection changed draws its maps
/// whole until it is unchanged for the settle time. `P` holds the light
/// type's projection inputs.
#[derive(Clone, Debug)]
pub(crate) struct LightSettlement<P> {
    identity: Option<P>,
    moved: bool,
    frame: u32,
    change: Duration,
}

impl<P> Default for LightSettlement<P> {
    fn default() -> Self {
        Self {
            identity: None,
            moved: false,
            frame: 0,
            change: Duration::ZERO,
        }
    }
}

impl<P: Clone + PartialEq> LightSettlement<P> {
    /// Notes what projects the light's shadow maps this frame, and returns
    /// whether the light is settled: unchanged since the frame before it
    /// first appeared, or, once it changed, for `settle_time`.
    pub(crate) fn settle(
        &mut self,
        identity: &P,
        frame: u32,
        now: Duration,
        settle_time: Duration,
    ) -> bool {
        if self.identity.as_ref() != Some(identity) {
            self.moved |= self.identity.is_some();
            self.identity = Some(identity.clone());
            self.frame = frame;
            self.change = now;
        }
        if self.moved {
            now.saturating_sub(self.change) >= settle_time
        } else {
            self.frame != frame
        }
    }
}

/// What the main world keeps for a spot light: whether it is cached (the
/// cache on and the light settled), and its one shadow view
/// ([`ShadowViewId::spot`]). Bevy adds and updates it on every [`SpotLight`].
#[derive(Component, Clone, Debug, Default)]
pub struct SpotLightShadowCache {
    /// Whether the light is cached this frame.
    pub(crate) cached: bool,
    pub(crate) settlement: LightSettlement<SpotShadowProjection>,
    pub(crate) view: ShadowViewCache,
}

impl SpotLightShadowCache {
    /// Whether the light is cached: shadows on, the cache on, and the light
    /// settled.
    pub fn is_cached(&self) -> bool {
        self.cached
    }

    /// The light's shadow view.
    pub fn view(&self) -> &ShadowViewCache {
        &self.view
    }

    /// See [`ShadowViewCache::static_entities`].
    pub fn static_entities(&self) -> &[Entity] {
        self.view.static_entities()
    }

    /// See [`ShadowViewCache::static_revision`].
    pub fn static_revision(&self) -> u64 {
        self.view.static_revision()
    }

    /// See [`ShadowViewCache::statics_walked`].
    pub fn statics_walked(&self) -> bool {
        self.view.statics_walked()
    }

    /// See [`ShadowViewCache::movers_tiles`].
    pub fn movers_tiles(&self) -> &ShadowTileMask {
        self.view.movers_tiles()
    }

    /// See [`ShadowViewCache::live_casters`].
    pub fn live_casters(&self) -> u32 {
        self.view.live_casters()
    }

    /// See [`ShadowViewCache::lights_a_view`].
    pub fn lights_a_view(&self) -> bool {
        self.view.lights_a_view()
    }

    /// See [`ShadowViewCache::single_view`].
    pub fn single_view(&self) -> bool {
        self.view.single_view()
    }
}

/// What the main world keeps for a directional light: whether it is cached
/// (the cache on, the light settled, and stable cascades,
/// [`StableCascades`](crate::StableCascades)), and the shadow view of each
/// stable cascade of each camera ([`ShadowSubview::Directional`]). Its other
/// cascades are drawn whole every frame. Bevy adds and updates it on every
/// [`DirectionalLight`](crate::DirectionalLight).
///
/// A stable cascade's view follows the rule of lights on its own projection:
/// one whose projection changed (it re-centered) is drawn whole, with no
/// cache work, until its projection is unchanged for
/// [`ShadowCacheSettings::settle_time`]; then it is cached again and drawn
/// anew once. At most one view of a light starts being cached in a frame, so
/// a middle and a far cascade never rebuild in the same frame.
#[derive(Component, Clone, Debug, Default)]
pub struct DirectionalLightShadowCache {
    /// Whether the light is cached this frame.
    pub(crate) cached: bool,
    pub(crate) settlement: LightSettlement<DirectionalLightSettle>,
    /// The stable cascades' views, by camera and cascade.
    pub(crate) views: HashMap<(Entity, u32), DirectionalViewState>,
    /// Views that started being cached this frame.
    pub(crate) entered: u32,
    /// The longest a settled view has waited for its turn, in frames.
    pub(crate) longest_wait: u32,
}

/// A stable cascade's view of a camera ([`DirectionalLightShadowCache`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct DirectionalViewState {
    /// Settles on the view's projection.
    pub(crate) settlement: LightSettlement<DirectionalShadowProjection>,
    /// The cached view, while it is cached.
    pub(crate) view: Option<ShadowViewCache>,
    /// The frame it settled and started waiting for its turn to be cached.
    pub(crate) waiting_since: Option<u32>,
}

/// Of views waiting for their turn (key, the frame each started waiting), the
/// one that waited longest (the first of equals).
pub(crate) fn longest_waiting<K: Copy>(views: impl IntoIterator<Item = (K, u32)>) -> Option<K> {
    let mut oldest: Option<(u32, K)> = None;
    for (key, since) in views {
        if oldest.is_none_or(|(first, _)| since < first) {
            oldest = Some((since, key));
        }
    }
    oldest.map(|(_, key)| key)
}

impl DirectionalLightShadowCache {
    /// Whether the light is cached: shadows on, the cache on, stable
    /// cascades, and the light settled. Each view is cached once its own
    /// projection settled ([`Self::view`]).
    pub fn is_cached(&self) -> bool {
        self.cached
    }

    /// A stable cascade's view of a camera, while it is cached.
    pub fn view(&self, camera: Entity, cascade: u32) -> Option<&ShadowViewCache> {
        self.views
            .get(&(camera, cascade))
            .and_then(|state| state.view.as_ref())
    }

    /// The cached views: (camera, cascade, view).
    pub fn views(&self) -> impl Iterator<Item = (Entity, u32, &ShadowViewCache)> {
        self.views.iter().filter_map(|((camera, cascade), state)| {
            state.view.as_ref().map(|view| (*camera, *cascade, view))
        })
    }

    /// How many views started being cached in this frame (at most one, see
    /// above). For tests.
    pub fn entered_this_frame(&self) -> u32 {
        self.entered
    }

    /// The longest a settled view of the light has waited this frame for its
    /// turn to be cached, in frames. For tests.
    pub fn longest_wait(&self) -> u32 {
        self.longest_wait
    }
}

/// What the render world reports about a cached shadow view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowCacheViewStatus {
    /// The [`ShadowViewCache::generation`] the report is about.
    pub generation: u64,
    /// The static revision whose shadow map is drawn and acknowledged: in the
    /// view's static layer, or, without a layer, in its shadow map.
    pub acknowledged_static_revision: Option<u64>,
    /// The view has no static layer it can use (the budget is spent, or the
    /// restore's blit isn't compiled yet): every frame in which something is
    /// live in its frustum draws it whole.
    pub fallback: bool,
    /// What the last frame did.
    pub state: ShadowCacheState,
    /// Texels the last frame restored from the static layer (whole tiles of a
    /// [`ShadowTileMask`] for live casters).
    pub restored_texels: u64,
}

/// What a frame did for a cached shadow view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShadowCacheState {
    /// Nothing: the shadow map is current.
    #[default]
    Kept,
    /// Restored the tiles live casters were drawn into and are in, then drew
    /// them.
    Movers,
    /// Drew the static layer, restored it whole, drew the live casters.
    Rebuild,
    /// Restored the whole static layer into a new place, drew the live
    /// casters.
    Restore,
    /// No static layer: drew everything.
    Fallback,
    /// Drew nothing: the light lights no view (its live casters stay where
    /// they were drawn until it lights a view again).
    Dormant,
}

/// How many frames a shadow view spent in each plan since it was first
/// cached, whatever its generation (for tools: a stand's `stat` or a log
/// line). The render world counts the plans of the frames whose passes were
/// planned; the main world counts the frames a stable directional cascade
/// was drawn whole while its projection settled (or waited for its turn to be
/// cached again).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowViewFrameCounts {
    /// [`ShadowCacheState::Kept`].
    pub kept: u64,
    /// [`ShadowCacheState::Movers`].
    pub movers: u64,
    /// [`ShadowCacheState::Restore`].
    pub restore: u64,
    /// [`ShadowCacheState::Rebuild`].
    pub rebuild: u64,
    /// [`ShadowCacheState::Fallback`].
    pub fallback: u64,
    /// [`ShadowCacheState::Dormant`].
    pub dormant: u64,
    /// Drawn whole, uncached, while its projection settled.
    pub settling: u64,
    /// Drawn whole with Bevy's projection: its stable window couldn't settle
    /// (the speed fallback, `StableCascadeProjections::is_stock`).
    pub stock: u64,
}

impl ShadowViewFrameCounts {
    /// Counts a frame planned so.
    pub fn add(&mut self, state: ShadowCacheState) {
        match state {
            ShadowCacheState::Kept => self.kept += 1,
            ShadowCacheState::Movers => self.movers += 1,
            ShadowCacheState::Restore => self.restore += 1,
            ShadowCacheState::Rebuild => self.rebuild += 1,
            ShadowCacheState::Fallback => self.fallback += 1,
            ShadowCacheState::Dormant => self.dormant += 1,
        }
    }
}

/// The render world's reports on cached shadow views, and every view's frame
/// counts ([`ShadowViewFrameCounts`]). Both worlds share it.
#[derive(Resource, Clone, Default)]
pub struct ShadowCacheFeedback(
    pub Arc<Mutex<HashMap<ShadowViewId, ShadowCacheViewStatus>>>,
    pub Arc<Mutex<HashMap<ShadowViewId, ShadowViewFrameCounts>>>,
);

impl ShadowCacheFeedback {
    /// A copy of every view's frame counts.
    pub fn frame_counts(&self) -> HashMap<ShadowViewId, ShadowViewFrameCounts> {
        self.1.lock().map(|map| map.clone()).unwrap_or_default()
    }

    /// A copy of the reports on every view.
    pub fn views(&self) -> HashMap<ShadowViewId, ShadowCacheViewStatus> {
        self.0.lock().map(|map| map.clone()).unwrap_or_default()
    }

    /// A copy of the reports on spot lights' views, by light.
    pub fn snapshot(&self) -> EntityHashMap<ShadowCacheViewStatus> {
        self.0
            .lock()
            .map(|map| {
                map.iter()
                    .filter(|(id, _)| id.subview == ShadowSubview::Spot)
                    .map(|(id, status)| (id.light, *status))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Registers what [`classify_shadow_casters`] needs besides its own system.
pub(crate) fn add_shadow_caster_tracking(app: &mut bevy_app::App) {
    app.register_required_components::<Mesh3d, ShadowCasterState>()
        .init_resource::<ShadowCasterTransitions>()
        .init_resource::<ShadowCacheCounters>()
        .init_resource::<ShadowCacheLoading>()
        .add_observer(on_remove_forced_live::<SkinnedMesh>)
        .add_observer(on_remove_forced_live::<MeshMorphWeights>)
        .add_observer(on_remove_forced_live::<MorphWeights>)
        .add_observer(on_remove_forced_live::<VisibilityRange>)
        .add_observer(on_remove_forced_live::<ShadowCasterLive>)
        .add_observer(on_remove_forced_live::<LiveShadowMaterial>)
        .add_observer(on_remove_forced_live::<NotShadowCaster>);
}

/// A caster that was forced live, or wasn't a caster, settles only after the
/// settle time.
fn on_remove_forced_live<C: Component>(
    remove: On<Remove<C>>,
    mut states: Query<&mut ShadowCasterState>,
    time: Option<Res<Time>>,
) {
    if let Ok(mut state) = states.get_mut(remove.entity) {
        let now = time.map_or(Duration::ZERO, |time| time.elapsed());
        state.reason = ShadowCasterChange::Forced;
        state.moved = true;
        state.last_change = now;
    }
}

/// Casters that settle after the settle time do so in waves, at most one per
/// this share of the settle time: casters that come to rest one after another
/// (debris after an explosion) settle together, and a light redraws once.
const SETTLE_WAVE_DIVISOR: u32 = 4;
/// The shadow views a wave may redraw, besides those of the caster that
/// waited longest; the casters beyond wait for the next wave.
const SETTLING_VIEWS_PER_WAVE: usize = 4;

/// The casters [`classify_shadow_casters`] can settle (meshes that cast a
/// shadow and nothing forces live), filtered by `F`.
type CasterQuery<'w, 's, F> = Query<
    'w,
    's,
    (
        Entity,
        Mut<'static, ShadowCasterState>,
        Ref<'static, GlobalTransform>,
        Option<Ref<'static, Aabb>>,
        Ref<'static, Mesh3d>,
        Ref<'static, InheritedVisibility>,
        Option<Ref<'static, RenderLayers>>,
        Option<Ref<'static, NoFrustumCulling>>,
        Option<Ref<'static, MeshTag>>,
        Option<Ref<'static, ShadowRepresentation>>,
    ),
    (
        Without<NotShadowCaster>,
        Without<SkinnedMesh>,
        Without<MeshMorphWeights>,
        Without<MorphWeights>,
        Without<VisibilityRange>,
        Without<ShadowCasterLive>,
        Without<LiveShadowMaterial>,
        F,
    ),
>;

/// The settled casters whose ticks say something was written: the candidates
/// of a change (Bevy marks writes, not changes).
type WrittenSettled = (
    With<StaticShadowCaster>,
    Or<(
        Changed<GlobalTransform>,
        Changed<Aabb>,
        Changed<Mesh3d>,
        Changed<InheritedVisibility>,
        Changed<RenderLayers>,
        Changed<NoFrustumCulling>,
        Changed<MeshTag>,
        Changed<ShadowCasterState>,
        Changed<ShadowRepresentation>,
    )>,
);

/// What decides a caster's shadow now.
fn signature_of(
    transform: &GlobalTransform,
    aabb: Option<&Aabb>,
    mesh: &Mesh3d,
    visibility: &InheritedVisibility,
    layers: Option<&RenderLayers>,
    no_frustum_culling: bool,
    tag: u32,
    representation: Option<&ShadowRepresentation>,
) -> CasterSignature {
    CasterSignature {
        world_from_local: transform.affine(),
        aabb: aabb.copied(),
        mesh: mesh.id(),
        tag,
        visible: visibility.get(),
        layers: layers.cloned().unwrap_or_default(),
        no_frustum_culling,
        representation: representation.copied(),
    }
}

/// Settled casters that must leave the static layers: no longer a casting
/// mesh, or forced live.
type LeavingFilter = (
    With<StaticShadowCaster>,
    Or<(
        Without<Mesh3d>,
        With<NotShadowCaster>,
        With<SkinnedMesh>,
        With<MeshMorphWeights>,
        With<MorphWeights>,
        With<VisibilityRange>,
        With<ShadowCasterLive>,
        With<LiveShadowMaterial>,
    )>,
);

/// Casting meshes forced live.
type ForcedLiveFilter = (
    With<Mesh3d>,
    Without<NotShadowCaster>,
    Or<(
        With<SkinnedMesh>,
        With<MeshMorphWeights>,
        With<MorphWeights>,
        With<VisibilityRange>,
        With<ShadowCasterLive>,
        With<LiveShadowMaterial>,
    )>,
);

/// An item of [`CasterQuery`].
type CasterItem<'w> = (
    Entity,
    Mut<'w, ShadowCasterState>,
    Ref<'w, GlobalTransform>,
    Option<Ref<'w, Aabb>>,
    Ref<'w, Mesh3d>,
    Ref<'w, InheritedVisibility>,
    Option<Ref<'w, RenderLayers>>,
    Option<Ref<'w, NoFrustumCulling>>,
    Option<Ref<'w, MeshTag>>,
    Option<Ref<'w, ShadowRepresentation>>,
);

/// What [`classify_shadow_casters`] keeps between frames.
#[derive(Default)]
#[doc(hidden)]
pub struct ClassifyState {
    frame: u32,
    /// For the `late_marker` fault: marker changes for the next frame.
    late: Vec<(Entity, bool)>,
    candidates: EntityHashSet,
    modified_meshes: HashSet<AssetId<Mesh>>,
    waiting: Vec<(Duration, Entity)>,
    /// When the next wave of settling casters may run.
    next_wave: Duration,
    /// Settled casters made live this frame.
    promoted: EntityHashSet,
    /// A wave's scratch: the cached views' bounds, which of them the wave
    /// redraws, the views a caster would add.
    cached_views: Vec<ShadowViewBounds>,
    in_wave: Vec<bool>,
    new_views: Vec<usize>,
}

/// Decides which casters are settled ([`StaticShadowCaster`]) this frame, by
/// the rule in the [module docs](self): after transform propagation and
/// bounds, before [`invalidate_shadow_caches`] and the lights'
/// visibility, so that the marker, the redrawn lights and the live casters'
/// lists agree in the frame that is extracted.
///
/// Its cost is a look at the change ticks of every settled caster (and a
/// comparison of the few whose ticks changed), and a look at every live
/// caster that can settle; casters forced live are skipped by archetype.
#[expect(clippy::too_many_arguments, reason = "a system")]
pub fn classify_shadow_casters(
    mut commands: Commands,
    (settings, loading): (Res<ShadowCacheSettings>, Option<Res<ShadowCacheLoading>>),
    time: Option<Res<Time>>,
    counters: ResMut<ShadowCacheCounters>,
    (mut reports, mut mesh_events): (
        MessageReader<InvalidateShadowCache>,
        MessageReader<AssetEvent<Mesh>>,
    ),
    (
        mut removed_layers,
        mut removed_no_culling,
        mut removed_aabbs,
        mut removed_tags,
        mut removed_representations,
    ): (
        RemovedComponents<RenderLayers>,
        RemovedComponents<NoFrustumCulling>,
        RemovedComponents<Aabb>,
        RemovedComponents<MeshTag>,
        RemovedComponents<ShadowRepresentation>,
    ),
    (mut live_casters, mut settled_casters): (
        CasterQuery<Without<StaticShadowCaster>>,
        ParamSet<(
            CasterQuery<WrittenSettled>,
            CasterQuery<With<StaticShadowCaster>>,
        )>,
    ),
    (leaving, forced, all_settled, settled_count): (
        Query<Entity, LeavingFilter>,
        Query<(), ForcedLiveFilter>,
        Query<Entity, With<StaticShadowCaster>>,
        Query<(), (With<StaticShadowCaster>, With<Mesh3d>)>,
    ),
    lights: Query<(
        &SpotLight,
        &GlobalTransform,
        &Frustum,
        &SpotLightShadowCache,
    )>,
    faults: Option<Res<ShadowCacheFaults>>,
    mut local: Local<ClassifyState>,
) {
    let ClassifyState {
        frame,
        late,
        candidates,
        modified_meshes,
        waiting,
        next_wave,
        promoted,
        cached_views,
        in_wave,
        new_views,
    } = &mut *local;
    *frame = frame.wrapping_add(1);
    let frame = *frame;
    let now = time.map_or(Duration::ZERO, |time| time.elapsed());
    // One write for the whole frame's counts.
    let counters = counters.into_inner();
    let late_marker = faults.as_ref().is_some_and(|faults| faults.late_marker);
    // The value a shader reads; the fault doesn't see it.
    let miss_tag_change = faults.as_ref().is_some_and(|faults| faults.miss_tag_change);
    let tag_of = |tag: Option<&MeshTag>| {
        if miss_tag_change {
            0
        } else {
            tag.map_or(0, |tag| tag.value)
        }
    };
    let loading = loading.is_none_or(|loading| loading.0);
    for (entity, settle) in late.drain(..) {
        set_settled(&mut commands, entity, settle);
    }

    candidates.clear();
    for report in reports.read() {
        if let InvalidateShadowCache::Caster(entity) = report {
            candidates.insert(*entity);
        }
    }
    let reported = candidates.clone();
    candidates.extend(
        removed_layers
            .read()
            .chain(removed_no_culling.read())
            .chain(removed_aabbs.read())
            .chain(removed_tags.read())
            .chain(removed_representations.read()),
    );
    modified_meshes.clear();
    // A mesh changed in place, or replaced under the same id (removed and
    // inserted again: `Added` on an id in use).
    modified_meshes.extend(mesh_events.read().filter_map(|event| match event {
        AssetEvent::Modified { id } | AssetEvent::Added { id } => Some(*id),
        _ => None,
    }));

    *counters = ShadowCacheCounters {
        live: 0,
        settling: 0,
        deferred: 0,
        fresh: 0,
        forced_live: 0,
        settled: 0,
        ..*counters
    };
    if !settings.enabled {
        // Nothing is cached: every caster is live, and the casters are seen
        // afresh when the cache is turned on again.
        for entity in &all_settled {
            set_settled(&mut commands, entity, false);
        }
        return;
    }
    counters.forced_live = forced.count() as u32;
    let mut mark = |commands: &mut Commands, entity: Entity, settle: bool| {
        if late_marker {
            late.push((entity, settle));
        } else {
            set_settled(commands, entity, settle);
        }
    };

    for entity in &leaving {
        mark(&mut commands, entity, false);
        counters.promotions += 1;
    }

    // Settled casters: only those whose ticks say something was written are
    // compared; a changed one is live again in this frame (its signature stays
    // as it settled until the marker's hook read it).
    counters.settled = settled_count.count() as u32;
    promoted.clear();
    let mut promote = |commands: &mut Commands,
                       entity: Entity,
                       state: &mut ShadowCasterState,
                       change: ShadowCasterChange| {
        state.change(change, frame, now);
        state.moved = true;
        mark(commands, entity, false);
    };
    for (
        entity,
        mut state,
        transform,
        aabb,
        mesh,
        visibility,
        layers,
        no_frustum_culling,
        tag,
        representation,
    ) in &mut settled_casters.p0()
    {
        let state = state.bypass_change_detection();
        let pending = state.pending.take();
        let signature = signature_of(
            &transform,
            aabb.as_deref(),
            &mesh,
            &visibility,
            layers.as_deref(),
            no_frustum_culling.is_some(),
            tag_of(tag.as_deref()),
            representation.as_deref(),
        );
        let change = state
            .signature
            .as_ref()
            .map_or(Some(ShadowCasterChange::Bounds), |old| {
                old.change_from(&signature)
            })
            .or(pending);
        if let Some(change) = change {
            promote(&mut commands, entity, state, change);
            promoted.insert(entity);
        }
    }
    // Reported or with a component removed (no tick shows it), or with a
    // modified mesh asset.
    if !candidates.is_empty() || !modified_meshes.is_empty() {
        let mut all = settled_casters.p1();
        let mut check = |commands: &mut Commands, item: CasterItem<'_>| {
            let (
                entity,
                mut state,
                transform,
                aabb,
                mesh,
                visibility,
                layers,
                no_frustum_culling,
                tag,
                representation,
            ) = item;
            let flagged = if reported.contains(&entity) {
                Some(ShadowCasterChange::Reported)
            } else if modified_meshes.contains(&mesh.id()) {
                Some(ShadowCasterChange::Mesh)
            } else {
                None
            };
            if promoted.contains(&entity) {
                return;
            }
            let signature = signature_of(
                &transform,
                aabb.as_deref(),
                &mesh,
                &visibility,
                layers.as_deref(),
                no_frustum_culling.is_some(),
                tag_of(tag.as_deref()),
                representation.as_deref(),
            );
            let state = state.bypass_change_detection();
            let change = flagged.or_else(|| {
                state
                    .signature
                    .as_ref()
                    .map_or(Some(ShadowCasterChange::Bounds), |old| {
                        old.change_from(&signature)
                    })
            });
            if let Some(change) = change {
                promote(commands, entity, state, change);
                promoted.insert(entity);
            }
        };
        if modified_meshes.is_empty() {
            for &entity in candidates.iter() {
                if let Ok(item) = all.get_mut(entity) {
                    check(&mut commands, item);
                }
            }
        } else {
            for item in &mut all {
                if candidates.contains(&item.0) || modified_meshes.contains(&item.4.id()) {
                    check(&mut commands, item);
                }
            }
        }
    }
    let promotions = promoted.len() as u32;
    counters.promotions += promotions as u64;
    counters.live += promotions;
    counters.settling += promotions;
    counters.settled -= promotions.min(counters.settled);

    // Live casters, every frame: their changes, and when they settle.
    waiting.clear();
    for (
        entity,
        mut state,
        transform,
        aabb,
        mesh,
        visibility,
        layers,
        no_frustum_culling,
        tag,
        representation,
    ) in &mut live_casters
    {
        let moved_tick = transform.is_changed();
        let other_ticks = aabb.as_ref().is_some_and(Ref::is_changed)
            || mesh.is_changed()
            || visibility.is_changed()
            || layers.as_ref().is_some_and(Ref::is_changed)
            || no_frustum_culling.as_ref().is_some_and(Ref::is_changed)
            || tag.as_ref().is_some_and(Ref::is_changed)
            || representation.as_ref().is_some_and(Ref::is_changed)
            || state.is_changed()
            || (!candidates.is_empty() && candidates.contains(&entity))
            || (!modified_meshes.is_empty() && modified_meshes.contains(&mesh.id()));
        let ticks = moved_tick || other_ticks;
        let signature = || {
            signature_of(
                &transform,
                aabb.as_deref(),
                &mesh,
                &visibility,
                layers.as_deref(),
                no_frustum_culling.is_some(),
                tag_of(tag.as_deref()),
                representation.as_deref(),
            )
        };
        let state = state.bypass_change_detection();
        let pending = state.pending.take();
        let flagged = if !ticks {
            None
        } else if reported.contains(&entity) {
            Some(ShadowCasterChange::Reported)
        } else if modified_meshes.contains(&mesh.id()) {
            Some(ShadowCasterChange::Mesh)
        } else {
            pending
        };

        counters.live += 1;
        // Bounds are known once the mesh is loaded (or it is drawn
        // everywhere): it can't settle before.
        if aabb.is_none() && no_frustum_culling.is_none() {
            continue;
        }
        let Some(old) = &state.signature else {
            state.signature = Some(signature());
            state.observed = frame;
            state.change(ShadowCasterChange::New, frame, now);
            if loading {
                counters.fresh += 1;
            } else {
                // A spawn after loading waits the settle time.
                state.moved = true;
                counters.settling += 1;
            }
            continue;
        };
        // A caster not seen live in the last frame (it was settled, forced
        // live, or wasn't a caster) is compared whatever its ticks say.
        let seen_last_frame = state.observed.wrapping_add(1) == frame;
        let mut change = None;
        if other_ticks || !seen_last_frame {
            let current = signature();
            change = old.change_from(&current);
            if change.is_some() {
                state.signature = Some(current);
            }
        } else if moved_tick {
            // Only the transform was written (a mover's frame): its bits
            // alone, in place.
            let world_from_local = transform.affine();
            if !same_bits(
                &world_from_local.to_cols_array(),
                &old.world_from_local.to_cols_array(),
            ) && let Some(signature) = state.signature.as_mut()
            {
                signature.world_from_local = world_from_local;
                change = Some(ShadowCasterChange::Transform);
            }
        }
        if let Some(change) = change.or(flagged) {
            state.change(change, frame, now);
        }
        state.observed = frame;

        if !state.moved {
            // Unchanged since it appeared: it settles in the next frame.
            if state.changed_frame != frame {
                mark(&mut commands, entity, true);
                counters.first_settles += 1;
            } else {
                counters.fresh += 1;
            }
        } else if now.saturating_sub(state.last_change) >= settings.settle_time {
            waiting.push((state.last_change, entity));
        } else {
            counters.settling += 1;
        }
    }

    if waiting.is_empty() {
        return;
    }
    if now < *next_wave {
        counters.deferred += waiting.len() as u32;
        return;
    }
    // A wave: the casters that waited longest first, while the views they
    // touch stay within the wave's budget.
    waiting.sort_unstable();
    cached_views.clear();
    cached_views.extend(lights.iter().filter(|(.., cache)| cache.cached).filter_map(
        |(spot_light, transform, frustum, cache)| {
            Some(ShadowViewBounds::spot(
                spot_light,
                transform,
                frustum,
                cache.view.projection.as_ref()?.render_layers().clone(),
            ))
        },
    ));
    in_wave.clear();
    in_wave.resize(cached_views.len(), false);
    let mut wave_views = 0;
    let mut settled_any = false;
    for &(_, entity) in waiting.iter() {
        let Ok((_, state, ..)) = live_casters.get(entity) else {
            continue;
        };
        let Some(signature) = &state.signature else {
            continue;
        };
        new_views.clear();
        new_views.extend(
            cached_views
                .iter()
                .enumerate()
                .filter(|(index, bounds)| !in_wave[*index] && signature.touches(bounds))
                .map(|(index, _)| index),
        );
        if settled_any && wave_views + new_views.len() > SETTLING_VIEWS_PER_WAVE {
            counters.deferred += 1;
            continue;
        }
        // The longest-waiting caster's views are the wave's own; the
        // budget counts the views the others add.
        if settled_any {
            wave_views += new_views.len();
        }
        for &index in new_views.iter() {
            in_wave[index] = true;
        }
        settled_any = true;
        mark(&mut commands, entity, true);
        counters.demotions += 1;
    }
    *next_wave = now + settings.settle_time / SETTLE_WAVE_DIVISOR;
}

fn set_settled(commands: &mut Commands, entity: Entity, settle: bool) {
    let Ok(mut entity_commands) = commands.get_entity(entity) else {
        return;
    };
    if settle {
        entity_commands.try_insert(StaticShadowCaster);
    } else {
        entity_commands.try_remove::<StaticShadowCaster>();
    }
}

/// Marks the cached shadow views whose static layer must be drawn again: a
/// caster settled into it or left it (its footprint, from
/// [`StaticShadowCaster`]'s hooks, in the view's frustum), an
/// [`InvalidateShadowCache`] names the view's light, or the budget or the
/// shadow map size changed. The light's visibility system
/// ([`check_point_light_mesh_visibility`] for spot lights) then bumps the
/// view's revision and finds its settled casters again.
///
/// [`check_point_light_mesh_visibility`]: crate::check_point_light_mesh_visibility
pub fn invalidate_shadow_caches(
    settings: Res<ShadowCacheSettings>,
    shadow_map: Res<DirectionalLightShadowMap>,
    mut invalidations: MessageReader<InvalidateShadowCache>,
    mut transitions: ResMut<ShadowCasterTransitions>,
    mut counters: ResMut<ShadowCacheCounters>,
    mut lights: Query<(
        Entity,
        Mut<SpotLightShadowCache>,
        &SpotLight,
        &GlobalTransform,
        &Frustum,
    )>,
    mut directional_lights: Query<(Entity, Mut<DirectionalLightShadowCache>, &CascadesFrusta)>,
    mut redraw_key: Local<Option<(u64, usize)>>,
    faults: Option<Res<ShadowCacheFaults>>,
) {
    let key = (settings.budget_bytes, shadow_map.size);
    let mut all = redraw_key.replace(key).is_some_and(|old| old != key);
    let mut invalidated_lights = Vec::new();
    for invalidation in invalidations.read() {
        match *invalidation {
            InvalidateShadowCache::Caster(_) => {}
            InvalidateShadowCache::Light(entity) => invalidated_lights.push(entity),
            InvalidateShadowCache::All => all = true,
        }
    }
    let transitions = &mut *transitions;
    if faults
        .as_ref()
        .is_some_and(|faults| faults.miss_static_invalidation)
    {
        transitions.left.clear();
    }
    if faults
        .as_ref()
        .is_some_and(|faults| faults.miss_settle_invalidation)
    {
        transitions.settled.clear();
    }
    if transitions.settled.is_empty()
        && transitions.left.is_empty()
        && invalidated_lights.is_empty()
        && !all
    {
        return;
    }

    for (light_entity, mut cache, spot_light, transform, frustum) in &mut lights {
        if !cache.cached || cache.view.invalidated {
            continue;
        }
        let touched = all
            || invalidated_lights.contains(&light_entity)
            || cache.view.projection.as_ref().is_none_or(|projection| {
                let bounds = ShadowViewBounds::spot(
                    spot_light,
                    transform,
                    frustum,
                    projection.render_layers().clone(),
                );
                transitions
                    .settled
                    .iter()
                    .chain(&transitions.left)
                    .any(|signature| {
                        signature
                            .as_ref()
                            .is_none_or(|signature| signature.touches(&bounds))
                    })
            });
        if touched {
            cache.view.invalidated = true;
            counters.rebuild_requests += 1;
        }
    }
    // A directional light's cached far cascades: their frusta without a near
    // plane, by camera.
    for (light_entity, mut cache, frusta) in &mut directional_lights {
        if !cache.cached {
            continue;
        }
        let light_invalidated = all || invalidated_lights.contains(&light_entity);
        for ((camera, cascade), state) in cache.bypass_change_detection().views.iter_mut() {
            let cascade = *cascade as usize;
            let Some(view) = state.view.as_mut() else {
                continue;
            };
            if view.invalidated {
                continue;
            }
            let touched = light_invalidated
                || view.projection.as_ref().is_none_or(|projection| {
                    let Some(frustum) = frusta
                        .frusta
                        .get(camera)
                        .and_then(|frusta| frusta.get(cascade))
                    else {
                        return true;
                    };
                    let stand_ins = match projection {
                        ShadowViewProjection::Directional(projection) => projection.stand_ins(),
                        ShadowViewProjection::Spot(_) => None,
                    };
                    let bounds = ShadowViewBounds::directional(
                        frustum,
                        projection.render_layers().clone(),
                        stand_ins,
                    );
                    transitions
                        .settled
                        .iter()
                        .chain(&transitions.left)
                        .any(|signature| {
                            signature
                                .as_ref()
                                .is_none_or(|signature| signature.touches(&bounds))
                        })
                });
            if touched {
                view.invalidated = true;
                counters.rebuild_requests += 1;
            }
        }
    }
    transitions.settled.clear();
    transitions.left.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn spot_clip_from_world(near_z: f32) -> Mat4 {
        // A light at the origin looking down -Z with a 90° cone.
        let transform = GlobalTransform::IDENTITY;
        spot_light_clip_from_view(core::f32::consts::FRAC_PI_4, near_z)
            * Mat4::from(spot_light_world_from_view(&transform).inverse())
    }

    /// Whether every vertex of `aabb` (placed by `world_from_local`) moved
    /// by up to `displacement` in world space stays inside `padded`.
    fn holds_displaced_vertices(
        aabb: &Aabb,
        padded: &Aabb,
        world_from_local: &Affine3A,
        displacement: f32,
    ) -> bool {
        let local_from_world = world_from_local.inverse();
        let mut directions = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let direction = Vec3A::new(x as f32, y as f32, z as f32);
                    if direction != Vec3A::ZERO {
                        directions.push(direction.normalize() * displacement);
                    }
                }
            }
        }
        (0..8).all(|corner| {
            let sign = |bit: u32| if corner & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let local = aabb.center + aabb.half_extents * Vec3A::new(sign(0), sign(1), sign(2));
            let world = world_from_local.transform_point3a(local);
            directions.iter().all(|direction| {
                let back = local_from_world.transform_point3a(world + *direction);
                ((back - padded.center).abs() - padded.half_extents)
                    .cmple(Vec3A::splat(1e-4))
                    .all()
            })
        })
    }

    #[test]
    fn shadow_view_ids() {
        let light = Entity::from_raw_u32(7).unwrap();
        let camera = Entity::from_raw_u32(9).unwrap();
        let spot = ShadowViewId::spot(light);
        assert_eq!((spot.camera(), spot.subview_index()), (None, 0));
        let cascade = ShadowViewId {
            light,
            subview: ShadowSubview::Directional { camera, cascade: 2 },
        };
        assert_eq!(
            (cascade.camera(), cascade.subview_index()),
            (Some(camera), 2)
        );
        let face = ShadowViewId {
            light,
            subview: ShadowSubview::PointFace(5),
        };
        assert_eq!((face.camera(), face.subview_index()), (None, 5));
        assert!(spot != cascade && cascade != face);
    }

    #[test]
    fn the_view_that_waited_longest_is_cached_first() {
        // Two views settle every other frame (a camera that keeps
        // re-centering), a third is settled all the time; one view is cached a
        // frame. Taking the first waiting view in walk order would starve the
        // third; the longest waiting one gets its turn at once.
        let mut since: [Option<u32>; 3] = [None; 3];
        let mut cached = [false; 3];
        let mut first_cached = None;
        for frame in 0..40u32 {
            for view in 0..2 {
                // Re-centered: drawn whole again, its wait starts over.
                if (frame + view as u32) % 2 == 0 {
                    cached[view] = false;
                    since[view] = None;
                }
            }
            let waiting = (0..3)
                .filter(|view| !cached[*view] && ((frame + *view as u32) % 2 == 1 || *view == 2))
                .map(|view| (view, *since[view].get_or_insert(frame)));
            if let Some(view) = longest_waiting(waiting.collect::<Vec<_>>()) {
                cached[view] = true;
                since[view] = None;
                if view == 2 && first_cached.is_none() {
                    first_cached = Some(frame);
                }
            }
        }
        assert!(
            first_cached.is_some_and(|frame| frame <= 2),
            "{first_cached:?}"
        );
        assert_eq!(longest_waiting([(7, 5), (8, 3), (9, 3)]), Some(8));
        assert_eq!(longest_waiting(core::iter::empty::<(u8, u32)>()), None);
    }

    #[test]
    fn lights_settle_by_the_rule() {
        let settle_time = Duration::from_secs(1);
        let at = Duration::from_secs_f32;
        let mut settlement = LightSettlement::<u32>::default();
        // Unchanged since the frame before it first appeared.
        assert!(!settlement.settle(&1, 1, at(0.0), settle_time));
        assert!(settlement.settle(&1, 2, at(0.016), settle_time));
        // Changed: it waits the settle time, from its last change.
        assert!(!settlement.settle(&2, 3, at(0.1), settle_time));
        assert!(!settlement.settle(&2, 4, at(0.6), settle_time));
        assert!(!settlement.settle(&3, 5, at(0.7), settle_time));
        assert!(!settlement.settle(&3, 6, at(1.5), settle_time));
        assert!(settlement.settle(&3, 7, at(1.7), settle_time));
    }

    #[test]
    fn reports_on_another_generation_are_ignored() {
        let id = ShadowViewId::spot(Entity::from_raw_u32(3).unwrap());
        let mut view = ShadowViewCache::default();
        assert_eq!(view.generation(), 0);
        view.begin();
        let generation = view.generation();
        assert_ne!(generation, 0);
        view.begin();
        assert_eq!(view.generation(), generation, "begun once");
        let mut statuses = HashMap::default();
        statuses.insert(
            id,
            ShadowCacheViewStatus {
                generation,
                acknowledged_static_revision: Some(1),
                ..Default::default()
            },
        );
        assert!(view.status(&statuses, &id).is_some());
        // The state created anew (its light got its cache component again).
        let mut recreated = ShadowViewCache::default();
        recreated.begin();
        assert_ne!(recreated.generation(), generation);
        assert!(recreated.status(&statuses, &id).is_none());
    }

    #[test]
    fn padding_holds_the_displacement_under_shear() {
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(0.5),
        };
        let live = LiveShadowMaterial {
            max_displacement: 1.0,
        };
        // Columns (1, 0, 0), (1, 0.1, 0), (0, 0, 1): every column at least 1
        // long, but a world unit along Y is 10 local units along local Y.
        let sheared = Affine3A::from_mat3(bevy_math::Mat3::from_cols(
            Vec3::X,
            Vec3::new(1.0, 0.1, 0.0),
            Vec3::Z,
        ));
        let padded = crate::padded_aabb(&aabb, &sheared, Some(&live)).unwrap();
        assert!(holds_displaced_vertices(&aabb, &padded, &sheared, 1.0));
        // Scaled and rotated too.
        let scaled = Affine3A::from_scale_rotation_translation(
            Vec3::new(0.2, 3.0, 1.0),
            bevy_math::Quat::from_rotation_y(0.7),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let padded = crate::padded_aabb(&aabb, &scaled, Some(&live)).unwrap();
        assert!(holds_displaced_vertices(&aabb, &padded, &scaled, 1.0));
        // A singular or nonfinite transform has no bounded box.
        let flat = Affine3A::from_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(crate::padded_aabb(&aabb, &flat, Some(&live)).is_none());
        let nan = Affine3A::from_scale(Vec3::new(1.0, f32::NAN, 1.0));
        assert!(crate::padded_aabb(&aabb, &nan, Some(&live)).is_none());
        // Without a live material the bounds are as they are.
        assert_eq!(crate::padded_aabb(&aabb, &flat, None), Some(aabb));
    }

    #[test]
    fn tile_masks() {
        const T: u32 = ShadowTileMask::TILE;
        let runs = |mask: &ShadowTileMask| {
            let mut runs = Vec::new();
            mask.for_each_run(|rect| runs.push(rect));
            runs
        };

        // A footprint sets the tiles of its texels with a margin of one
        // texel: here texels 255..513 and 511..769 of a 1024 map.
        let mut mask = ShadowTileMask::new(1024);
        assert!(mask.is_empty());
        mask.insert(ShadowFootprint::Rect {
            min: Vec2::new(0.25, 0.5),
            max: Vec2::new(0.5, 0.75),
        });
        let (x0, x1, y0, y1) = (255 / T, 512 / T + 1, 511 / T, 768 / T + 1);
        let expected: Vec<URect> = (y0..y1)
            .map(|row| URect::new(x0 * T, row * T, x1 * T, (row + 1) * T))
            .collect();
        assert_eq!(runs(&mask), expected);
        assert_eq!(
            mask.texels(),
            ((x1 - x0) * (y1 - y0)) as u64 * (T * T) as u64
        );

        // Clamped to the map, the last tile clamped to its edge.
        let mut corner = ShadowTileMask::new(1000);
        corner.insert(ShadowFootprint::Rect {
            min: Vec2::new(0.999, 0.999),
            max: Vec2::ONE,
        });
        let last = 998 / T * T;
        assert_eq!(runs(&corner), vec![URect::new(last, last, 1000, 1000)]);
        assert_eq!(corner.texels(), (1000 - last) as u64 * (1000 - last) as u64);

        // Nowhere and anywhere.
        let mut whole = ShadowTileMask::new(1000);
        whole.insert(ShadowFootprint::Empty);
        assert!(whole.is_empty());
        assert_eq!(whole.coverage(), 0.0);
        whole.insert(ShadowFootprint::Whole);
        assert_eq!(whole.texels(), 1000 * 1000);
        assert_eq!(whole.coverage(), 1.0);
        assert!(mask.coverage() > 0.0 && mask.coverage() < 0.2);

        // A run of tiles across two words of a row is one run.
        let mut wide = ShadowTileMask::new(128 * T);
        wide.set_tiles(3..4, 60, 69);
        wide.set_tiles(3..4, 72, 72);
        assert_eq!(
            runs(&wide),
            vec![
                URect::new(60 * T, 3 * T, 70 * T, 4 * T),
                URect::new(72 * T, 3 * T, 73 * T, 4 * T)
            ]
        );

        // A run per row and set tiles in a row; runs in rows below aren't
        // merged (one quad per run was faster on the Intel GPU).
        let mut stack = ShadowTileMask::new(128 * T);
        stack.set_tiles(0..2, 0, 1);
        stack.set_tiles(2..3, 0, 2);
        stack.set_tiles(1..3, 5, 5);
        assert_eq!(
            runs(&stack),
            vec![
                URect::new(0, 0, 2 * T, T),
                URect::new(0, T, 2 * T, 2 * T),
                URect::new(5 * T, T, 6 * T, 2 * T),
                URect::new(0, 2 * T, 3 * T, 3 * T),
                URect::new(5 * T, 2 * T, 6 * T, 3 * T)
            ]
        );
        assert_eq!(stack.texels(), 9 * (T * T) as u64);

        // A union; a mask of another size covers everything.
        let mut union = mask.clone();
        union.union_with(&corner);
        let mut full = ShadowTileMask::new(1024);
        full.fill();
        assert_eq!(union, full);
        let mut union = ShadowTileMask::new(1024);
        union.union_with(&ShadowTileMask::new(2048));
        assert!(union.is_empty());
        let mut other = ShadowTileMask::new(1024);
        other.set_tiles(0..1, 0, 0);
        union.union_with(&other);
        assert_eq!(union.texels(), (T * T) as u64);
        union.union_with(&mask);
        assert_eq!(union.texels(), mask.texels() + (T * T) as u64);

        // The fault drops a tile that isn't set now, if there is one.
        let mut dropped = union.clone();
        dropped.drop_one_tile(&mask);
        assert_eq!(dropped, mask);
        let mut dropped = mask.clone();
        dropped.drop_one_tile(&mask);
        assert_eq!(dropped.texels(), mask.texels() - (T * T) as u64);
    }

    #[test]
    fn footprint_of_a_box_in_front_of_the_light() {
        let clip_from_world = spot_clip_from_world(0.1);
        // A unit box 10 m in front of the light, centered on its axis.
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(0.5),
        };
        let world_from_local = Affine3A::from_translation(Vec3::new(0.0, 0.0, -10.0));
        let ShadowFootprint::Rect { min, max } =
            ShadowFootprint::of_box(&clip_from_world, 0.1, &aabb, &world_from_local)
        else {
            panic!("expected a rectangle");
        };
        // The near face at 9.5 m spans ±0.5 / 9.5 of the tan(45°) half-width.
        let half = 0.5 / 9.5 * 0.5;
        assert!((min.x - (0.5 - half)).abs() < 1e-5, "{min:?}");
        assert!((max.x - (0.5 + half)).abs() < 1e-5, "{max:?}");
        assert!((min.y - (0.5 - half)).abs() < 1e-5, "{min:?}");
        assert!((max.y - (0.5 + half)).abs() < 1e-5, "{max:?}");

        // Off to the right (+X) the box lands right in the map (larger u).
        let right = Affine3A::from_translation(Vec3::new(5.0, 0.0, -10.0));
        let ShadowFootprint::Rect { min, .. } =
            ShadowFootprint::of_box(&clip_from_world, 0.1, &aabb, &right)
        else {
            panic!("expected a rectangle");
        };
        assert!(min.x > 0.6);
        // Up (+Y) lands at the top of the map (smaller v).
        let up = Affine3A::from_translation(Vec3::new(0.0, 5.0, -10.0));
        let ShadowFootprint::Rect { max, .. } =
            ShadowFootprint::of_box(&clip_from_world, 0.1, &aabb, &up)
        else {
            panic!("expected a rectangle");
        };
        assert!(max.y < 0.4);
    }

    #[test]
    fn footprint_of_a_box_across_the_near_plane_is_whole() {
        let clip_from_world = spot_clip_from_world(0.1);
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(0.5),
        };
        // Spans 0.4 m behind to 0.6 m in front of the light.
        let world_from_local = Affine3A::from_translation(Vec3::new(0.0, 0.0, -0.1));
        assert_eq!(
            ShadowFootprint::of_box(&clip_from_world, 0.1, &aabb, &world_from_local),
            ShadowFootprint::Whole
        );
    }

    #[test]
    fn footprint_is_clamped_to_the_map() {
        let clip_from_world = spot_clip_from_world(0.1);
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::new(20.0, 0.5, 0.5),
        };
        let world_from_local = Affine3A::from_translation(Vec3::new(0.0, 0.0, -10.0));
        let ShadowFootprint::Rect { min, max } =
            ShadowFootprint::of_box(&clip_from_world, 0.1, &aabb, &world_from_local)
        else {
            panic!("expected a rectangle");
        };
        assert_eq!(min.x, 0.0);
        assert_eq!(max.x, 1.0);
    }

    #[test]
    fn movers_grid_finds_every_caster_near_a_light() {
        // Casters on a pseudo-random spread, some wide or without bounds.
        let mut casters = Vec::new();
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for index in 0..2000u32 {
            let center = Vec3A::new(
                next() * 200.0 - 100.0,
                next() * 10.0,
                next() * 200.0 - 100.0,
            );
            let half_extents = Vec3A::splat(if index % 50 == 0 { 5.0 } else { next() * 2.0 });
            let bounds = (index % 333 != 0).then_some((center, half_extents));
            casters.push((Entity::from_raw_u32(index + 1).unwrap(), bounds));
        }
        let mut grid = MoversGrid::default();
        grid.build(casters.iter().copied());

        for (light, radius) in [
            (Vec3A::new(0.0, 8.0, 0.0), 20.0),
            (Vec3A::new(-97.0, 3.0, 55.0), 6.0),
            (Vec3A::new(31.3, -2.0, -12.9), 0.5),
            (Vec3A::new(0.0, 0.0, 0.0), 1.0e9),
        ] {
            let mut candidates = Vec::new();
            grid.candidates(light, radius, &mut candidates);
            for (entity, bounds) in &casters {
                // What the light's sphere test can accept: a caster whose
                // bounds reach the sphere, or that has none.
                let reaches = bounds.is_none_or(|(center, half_extents)| {
                    let closest = light.clamp(center - half_extents, center + half_extents);
                    closest.distance(light) <= radius
                });
                if reaches {
                    assert!(candidates.contains(entity), "{entity:?} near {light:?}");
                }
            }
            candidates.sort_unstable();
            let count = candidates.len();
            candidates.dedup();
            assert_eq!(count, candidates.len(), "no caster twice");
        }
    }

    fn classify_app() -> bevy_app::App {
        let mut app = bevy_app::App::new();
        app.add_message::<InvalidateShadowCache>()
            .add_message::<AssetEvent<Mesh>>()
            .insert_resource(ShadowCacheSettings {
                enabled: true,
                settle_time: Duration::from_secs(1),
                ..Default::default()
            })
            .insert_resource(Time::<()>::default())
            .add_systems(bevy_app::Update, classify_shadow_casters);
        add_shadow_caster_tracking(&mut app);
        app
    }

    fn step(app: &mut bevy_app::App, seconds: f32) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(seconds));
        app.update();
    }

    #[test]
    fn casters_settle_by_the_rule() {
        let mut app = classify_app();
        let caster = app
            .world_mut()
            .spawn((
                Mesh3d::default(),
                GlobalTransform::from_xyz(1.0, 2.0, 3.0),
                Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let settled =
            |app: &bevy_app::App| app.world().entity(caster).contains::<StaticShadowCaster>();
        let counters = |app: &bevy_app::App| *app.world().resource::<ShadowCacheCounters>();

        // It appears live and settles in the next frame.
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(counters(&app).fresh, 1);
        step(&mut app, 0.016);
        assert!(settled(&app));
        assert_eq!(counters(&app).first_settles, 1);

        // The same transform written again changes nothing.
        let same = GlobalTransform::from_xyz(1.0, 2.0, 3.0);
        *app.world_mut().get_mut::<GlobalTransform>(caster).unwrap() = same;
        step(&mut app, 0.016);
        assert!(settled(&app));
        assert_eq!(counters(&app).promotions, 0);

        // One bit of motion is live in the same frame; still for the settle
        // time, it settles again.
        let moved = GlobalTransform::from_xyz(1.0f32.next_up(), 2.0, 3.0);
        *app.world_mut().get_mut::<GlobalTransform>(caster).unwrap() = moved;
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(counters(&app).promotions, 1);
        for _ in 0..5 {
            step(&mut app, 0.1);
            assert!(!settled(&app), "before the settle time");
        }
        step(&mut app, 0.7);
        assert!(settled(&app));
        assert_eq!(counters(&app).demotions, 1);

        // Forced live, and the settle time after it.
        app.world_mut().entity_mut(caster).insert(ShadowCasterLive);
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(counters(&app).forced_live, 1);
        app.world_mut()
            .entity_mut(caster)
            .remove::<ShadowCasterLive>();
        step(&mut app, 0.016);
        step(&mut app, 0.016);
        assert!(!settled(&app), "no longer forced: it waits the settle time");
        step(&mut app, 1.0);
        assert!(settled(&app));

        // A material change makes it live; a material noted the first time
        // doesn't.
        let material = UntypedAssetId::Uuid {
            type_id: core::any::TypeId::of::<Mesh>(),
            uuid: bevy_asset::uuid::Uuid::from_u128(1),
        };
        let other = UntypedAssetId::Uuid {
            type_id: core::any::TypeId::of::<Mesh>(),
            uuid: bevy_asset::uuid::Uuid::from_u128(2),
        };
        app.world_mut()
            .get_mut::<ShadowCasterState>(caster)
            .unwrap()
            .note_material(material);
        step(&mut app, 0.016);
        assert!(settled(&app));
        app.world_mut()
            .get_mut::<ShadowCasterState>(caster)
            .unwrap()
            .note_material(other);
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(
            app.world()
                .get::<ShadowCasterState>(caster)
                .unwrap()
                .reason(),
            ShadowCasterChange::Material
        );

        // Turned off, nothing stays settled.
        step(&mut app, 2.0);
        assert!(settled(&app));
        app.world_mut()
            .resource_mut::<ShadowCacheSettings>()
            .enabled = false;
        step(&mut app, 0.016);
        assert!(!settled(&app));
    }

    #[test]
    fn a_mesh_tag_change_makes_a_settled_caster_live() {
        let mut app = classify_app();
        let caster = app
            .world_mut()
            .spawn((
                Mesh3d::default(),
                GlobalTransform::IDENTITY,
                Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                InheritedVisibility::VISIBLE,
                MeshTag::new(1),
            ))
            .id();
        let settled =
            |app: &bevy_app::App| app.world().entity(caster).contains::<StaticShadowCaster>();
        let promotions =
            |app: &bevy_app::App| app.world().resource::<ShadowCacheCounters>().promotions;
        step(&mut app, 0.016);
        step(&mut app, 0.016);
        assert!(settled(&app));

        // The same tag written again changes nothing.
        *app.world_mut().get_mut::<MeshTag>(caster).unwrap() = MeshTag::new(1);
        step(&mut app, 0.016);
        assert!(settled(&app));
        assert_eq!(promotions(&app), 0);

        // Another tag (a shader may place the vertices by it) is live in the
        // same frame, as a mesh change.
        *app.world_mut().get_mut::<MeshTag>(caster).unwrap() = MeshTag::new(2);
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(promotions(&app), 1);
        assert_eq!(
            app.world()
                .get::<ShadowCasterState>(caster)
                .unwrap()
                .reason(),
            ShadowCasterChange::Mesh
        );
        // Settled before: it settles again the settle time after its first
        // frame live (which takes in the new signature).
        step(&mut app, 0.016);
        step(&mut app, 1.1);
        assert!(settled(&app));

        // Removed: the shaders read 0.
        app.world_mut().entity_mut(caster).remove::<MeshTag>();
        step(&mut app, 0.016);
        assert!(!settled(&app));
        assert_eq!(promotions(&app), 2);
        step(&mut app, 0.016);
        step(&mut app, 1.1);
        assert!(settled(&app));

        // The fault doesn't see it.
        app.insert_resource(ShadowCacheFaults {
            miss_tag_change: true,
            ..Default::default()
        });
        app.world_mut().entity_mut(caster).insert(MeshTag::new(3));
        step(&mut app, 0.016);
        assert!(settled(&app));
        assert_eq!(promotions(&app), 2);
    }

    #[test]
    fn a_spawn_after_loading_waits_the_settle_time() {
        let mut app = classify_app();
        app.insert_resource(ShadowCacheLoading(false));
        let caster = app
            .world_mut()
            .spawn((
                Mesh3d::default(),
                GlobalTransform::IDENTITY,
                Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                InheritedVisibility::VISIBLE,
            ))
            .id();
        step(&mut app, 0.016);
        step(&mut app, 0.5);
        assert!(!app.world().entity(caster).contains::<StaticShadowCaster>());
        // Despawned before the settle time: no static layer ever saw it.
        app.world_mut().despawn(caster);
        step(&mut app, 0.016);
        assert!(app
            .world()
            .resource::<ShadowCasterTransitions>()
            .left
            .is_empty());
        let counters = *app.world().resource::<ShadowCacheCounters>();
        assert_eq!(counters.first_settles + counters.demotions, 0);

        let caster = app
            .world_mut()
            .spawn((
                Mesh3d::default(),
                GlobalTransform::IDENTITY,
                Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                InheritedVisibility::VISIBLE,
            ))
            .id();
        step(&mut app, 0.016);
        step(&mut app, 1.1);
        assert!(app.world().entity(caster).contains::<StaticShadowCaster>());
    }

    #[test]
    fn a_caster_that_moved_waits_while_one_that_was_set_up_does_not() {
        let mut app = classify_app();
        let spawn = |app: &mut bevy_app::App| {
            app.world_mut()
                .spawn((
                    Mesh3d::default(),
                    GlobalTransform::IDENTITY,
                    Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                    InheritedVisibility::VISIBLE,
                ))
                .id()
        };
        let mover = spawn(&mut app);
        let hidden = spawn(&mut app);
        step(&mut app, 0.016);
        // Moved in its second frame: a mover, it waits the settle time.
        *app.world_mut().get_mut::<GlobalTransform>(mover).unwrap() =
            GlobalTransform::from_xyz(0.1, 0.0, 0.0);
        // Hidden in its second frame (set-up, not motion): it waits a frame.
        *app.world_mut()
            .get_mut::<InheritedVisibility>(hidden)
            .unwrap() = InheritedVisibility::HIDDEN;
        step(&mut app, 0.016);
        step(&mut app, 0.016);
        assert!(!app.world().entity(mover).contains::<StaticShadowCaster>());
        assert!(app.world().entity(hidden).contains::<StaticShadowCaster>());
        step(&mut app, 1.0);
        assert!(app.world().entity(mover).contains::<StaticShadowCaster>());
    }
}
