//! A debug check of the shadow cache (`bevy_light`'s `ShadowCacheSettings`)
//! against full redraws, texel by texel, view by view:
//! [`ShadowCacheAuditPlugin`]. It checks spot lights and point lights' faces
//! (with twins) and cached directional cascades (with reference views of
//! their exact projection).

use alloc::{borrow::Cow, sync::Arc};
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_camera::visibility::{NoFrustumCulling, RenderLayers, ViewVisibility};
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_light::{
    DirectionalLightShadowCache, PointLight, ShadowCacheExempt, ShadowCacheFaults, ShadowSubview,
    ShadowViewId, SpotLight,
};
use bevy_math::UVec2;
use bevy_render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};
use bevy_transform::TransformSystems;
use bevy_utils::default;

use crate::{
    acknowledge_shadow_caches, ExtractedPointLight, GlobalClusterableObjectMeta,
    PointAndSpotLightViewEntities, ShadowCachePlan, ShadowViewCaches, ShadowViewKey,
    ViewShadowBindings, CORE_3D_SHADOW_MAP_FORMAT,
};

/// Checks every cached spot light's shadow map and every cached point light's
/// faces against a full redraw, every frame. A debug tool: add it only to look for a fault of the cache, it
/// costs a shadow map per cached light and a compare pass and read-back per
/// frame.
///
/// Each shadowed spot light gets a twin: a child [`SpotLight`] with the same
/// projection, no intensity and no cache ([`ShadowCacheAuditTwin`], which the
/// cache leaves alone), so Bevy draws its shadow map from scratch every
/// frame, from its own list of casters, into its own layer of the same
/// texture. After the frame's passes a compute pass compares, texel by texel
/// as `u32` bits, each cached light's shadow map (the working layer) with its
/// twin's, and its static layer with the twin's and with the working layer.
/// The results arrive a few frames later in [`ShadowCacheAuditResults`].
///
/// A shadowed [`PointLight`] gets a twin the same way (a child point light
/// with its range, near plane and layers, no intensity, no cache), and each
/// of its six faces is compared with the twin's face (its layers
/// `light_index * 6 + face` of the point light texture), one record a face.
///
/// The twins change what the cache sees and does: their full views find every
/// caster in the light's frustum every frame, so its settled casters stay
/// visible, and their layers make the shadow map texture larger. What runs is
/// set at run time by [`ShadowCacheAuditSettings`] (on, with twins, by
/// default). Without twins the audit compares only each cached light's shadow
/// map with its static layer, or, with [`ShadowCacheAuditSettings::hashes`],
/// hashes the shadow map of every shadowed spot light, cached or not, for a
/// comparison with another run (one with the cache off draws every map from
/// scratch). While it's off it does nothing (twins already spawned stay).
///
/// A cached directional cascade (a camera's far cascade) has no twin: its
/// reference is a render-only view with the cascade's exact projection that
/// draws every caster the cascade draws, found anew each frame
/// (`ShadowCacheFaults::directional_reference`, which the plugin sets), into a
/// texture of its own; right after the camera's shadow passes, the cascade's
/// map is copied beside it, before another camera's cascades draw into the
/// same layer. The compare reads the two.
pub struct ShadowCacheAuditPlugin;

/// What [`ShadowCacheAuditPlugin`] does, changed at run time.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCacheAuditSettings {
    /// Compares every frame.
    pub enabled: bool,
    /// Gives every cached light a twin and compares with it.
    pub fresh_redraws: bool,
    /// Without twins: hashes the shadow map of every shadowed spot light
    /// ([`ShadowCacheAuditRecord::hash`]) instead of comparing cached ones
    /// with their static layers.
    pub hashes: bool,
}

impl Default for ShadowCacheAuditSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            fresh_redraws: true,
            hashes: false,
        }
    }
}

impl Plugin for ShadowCacheAuditPlugin {
    fn build(&self, app: &mut App) {
        let results = ShadowCacheAuditResults::default();
        app.insert_resource(results.clone())
            .init_resource::<ShadowCacheAuditSettings>()
            .add_systems(
                PostUpdate,
                (
                    (add_audit_twins, add_point_audit_twins)
                        .run_if(|settings: Res<ShadowCacheAuditSettings>| {
                            settings.enabled && settings.fresh_redraws
                        })
                        .before(TransformSystems::Propagate),
                    request_directional_references.before(TransformSystems::Propagate),
                ),
            );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(results)
            .init_resource::<AuditPairs>()
            .add_systems(ExtractSchedule, extract_audit_pairs)
            .add_systems(
                Render,
                audit_shadow_caches
                    .in_set(RenderSystems::Cleanup)
                    .before(acknowledge_shadow_caches),
            );
    }
}

/// The twin of a spot or point light (its parent), drawn from scratch every
/// frame by [`ShadowCacheAuditPlugin`]; never cached.
#[derive(Component, Clone, Copy, Debug)]
#[require(ShadowCacheExempt)]
pub struct ShadowCacheAuditTwin {
    /// The cached light.
    pub light: Entity,
}

/// A spot or point light's audit twin.
#[derive(Component)]
struct HasShadowCacheAuditTwin(Entity);

/// One differing texel of a working layer and its twin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCacheAuditTexel {
    /// The texel.
    pub texel: UVec2,
    /// The cached light's shadow map, as `f32` bits.
    pub working: u32,
    /// The full redraw, as `f32` bits.
    pub fresh: u32,
    /// The static layer, as `f32` bits (0 without one).
    pub static_layer: u32,
}

/// Whether an audited view-frame was compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCacheAuditStatus {
    /// Compared (or hashed).
    Compared,
    /// The light lights no view ([`ShadowCachePlan::Dormant`]): it drew
    /// nothing and nothing reads its map. Skipped.
    Dormant,
    /// Expected but not compared, and why: a map that wasn't drawn, or one
    /// drawn into another texture than the one lighting reads.
    Missing(&'static str),
}

/// One comparison of a cached shadow view with its full redraw.
#[derive(Clone, Debug)]
pub struct ShadowCacheAuditRecord {
    /// [`FrameCount`] when the frame was extracted.
    pub frame: u32,
    /// The view-frames the audit expected in this frame (one record each):
    /// with twins or hashes the view of every shadowed spot light the main
    /// world extracted, without them every cached view planned this frame.
    pub expected: u32,
    /// Whether it was compared.
    pub status: ShadowCacheAuditStatus,
    /// The audit's own count of rendered frames (an application may pin
    /// [`FrameCount`]).
    pub audit_frame: u32,
    /// The shadow view.
    pub view: ShadowViewId,
    /// Its light (main world).
    pub light: Entity,
    /// The shadow map layers of the light and its twin (the light's again
    /// without twins).
    pub layers: (u32, u32),
    /// Whether the shadow map was compared with a full redraw (a twin).
    pub fresh: bool,
    /// What the cache planned for the light this frame.
    pub plan: ShadowCachePlan,
    /// The static revision the main world asked for.
    pub static_revision: u64,
    /// The revision the static layer held before this frame.
    pub static_content: Option<u64>,
    /// The revision the shadow map held before this frame.
    pub working: Option<u64>,
    /// Whether the static view recorded its pass this frame.
    pub static_drawn: bool,
    /// The texels of the tiles movable casters touch now.
    pub movers_texels: u64,
    /// The texels movable casters were drawn into and not restored since,
    /// before this frame.
    pub written_texels: u64,
    /// The texels restored this frame.
    pub restored_texels: u64,
    /// Texels of the shadow map that differ from the full redraw.
    pub differing: u32,
    /// The first of them (smallest row, then column).
    pub first: Option<UVec2>,
    /// The largest difference, in units in the last place.
    pub max_ulps: u32,
    /// Up to 16 of the differing texels (in no particular order); without a
    /// twin, of the texels where the shadow map differs from its static
    /// layer.
    pub texels: Vec<ShadowCacheAuditTexel>,
    /// With a static layer: texels of it that differ from the full redraw
    /// (it holds only the static casters: equal only while no movable
    /// caster is in the light's frustum), the first, the largest difference.
    pub static_differing: Option<(u32, Option<UVec2>, u32)>,
    /// With a static layer: texels of the shadow map that differ from it.
    pub working_vs_static: Option<u32>,
    /// With [`ShadowCacheAuditSettings::hashes`]: a hash of the light's shadow
    /// map, its texels' bits and places.
    pub hash: Option<u64>,
    /// With [`ShadowCacheAuditSettings::hashes`]: whether the light was
    /// cached.
    pub cached: bool,
}

/// The audit's results, as they arrive from the GPU. Both worlds share it;
/// the application takes them.
#[derive(Resource, Clone, Default)]
pub struct ShadowCacheAuditResults(pub Arc<Mutex<Vec<ShadowCacheAuditRecord>>>);

impl ShadowCacheAuditResults {
    /// Takes the results that arrived.
    pub fn take(&self) -> Vec<ShadowCacheAuditRecord> {
        self.0
            .lock()
            .map(|mut records| core::mem::take(&mut *records))
            .unwrap_or_default()
    }
}

/// Gives every shadowed spot light a twin and keeps what decides the twin's
/// shadow map the light's: its projection and its render layers (the
/// transform is the parent's).
fn add_audit_twins(
    mut commands: Commands,
    lights: Query<
        (
            Entity,
            &SpotLight,
            Option<&RenderLayers>,
            Option<&HasShadowCacheAuditTwin>,
        ),
        (Without<ShadowCacheAuditTwin>, Without<ShadowCacheExempt>),
    >,
    mut twins: Query<(&mut SpotLight, Option<&RenderLayers>), With<ShadowCacheAuditTwin>>,
) {
    for (entity, spot_light, layers, twin) in &lights {
        // What the shadow map depends on; the twin lights nothing.
        let wanted = SpotLight {
            intensity: 0.0,
            contact_shadows_enabled: false,
            ..*spot_light
        };
        let projection = |light: &SpotLight| {
            (
                light.shadow_maps_enabled,
                light.range,
                light.radius,
                light.shadow_map_near_z,
                light.outer_angle,
                light.inner_angle,
            )
        };
        match twin {
            Some(HasShadowCacheAuditTwin(twin_entity)) => {
                if let Ok((mut twin, twin_layers)) = twins.get_mut(*twin_entity) {
                    if projection(&twin) != projection(&wanted) {
                        *twin = wanted;
                    }
                    if twin_layers != layers {
                        match layers {
                            Some(layers) => commands.entity(*twin_entity).insert(layers.clone()),
                            None => commands.entity(*twin_entity).remove::<RenderLayers>(),
                        };
                    }
                }
            }
            None if spot_light.shadow_maps_enabled => {
                let mut twin = commands.spawn((
                    wanted,
                    ShadowCacheAuditTwin { light: entity },
                    NoFrustumCulling,
                ));
                if let Some(layers) = layers {
                    twin.insert(layers.clone());
                }
                let twin = twin.id();
                commands
                    .entity(entity)
                    .add_child(twin)
                    .insert(HasShadowCacheAuditTwin(twin));
            }
            None => {}
        }
    }
}

/// Gives every shadowed point light a twin and keeps what decides the twin's
/// shadow maps the light's: its range, near plane and render layers (its
/// translation is the parent's; Bevy ignores rotation and scale).
fn add_point_audit_twins(
    mut commands: Commands,
    lights: Query<
        (
            Entity,
            &PointLight,
            Option<&RenderLayers>,
            Option<&HasShadowCacheAuditTwin>,
        ),
        (Without<ShadowCacheAuditTwin>, Without<ShadowCacheExempt>),
    >,
    mut twins: Query<(&mut PointLight, Option<&RenderLayers>), With<ShadowCacheAuditTwin>>,
) {
    for (entity, point_light, layers, twin) in &lights {
        let wanted = PointLight {
            intensity: 0.0,
            contact_shadows_enabled: false,
            ..*point_light
        };
        let projection = |light: &PointLight| {
            (
                light.shadow_maps_enabled,
                light.range,
                light.radius,
                light.shadow_map_near_z,
            )
        };
        match twin {
            Some(HasShadowCacheAuditTwin(twin_entity)) => {
                if let Ok((mut twin, twin_layers)) = twins.get_mut(*twin_entity) {
                    if projection(&twin) != projection(&wanted) {
                        *twin = wanted;
                    }
                    if twin_layers != layers {
                        match layers {
                            Some(layers) => commands.entity(*twin_entity).insert(layers.clone()),
                            None => commands.entity(*twin_entity).remove::<RenderLayers>(),
                        };
                    }
                }
            }
            None if point_light.shadow_maps_enabled => {
                let mut twin = commands.spawn((
                    wanted,
                    ShadowCacheAuditTwin { light: entity },
                    NoFrustumCulling,
                ));
                if let Some(layers) = layers {
                    twin.insert(layers.clone());
                }
                let twin = twin.id();
                commands
                    .entity(entity)
                    .add_child(twin)
                    .insert(HasShadowCacheAuditTwin(twin));
            }
            None => {}
        }
    }
}

/// Asks the cached directional cascades for their reference lists while the
/// audit compares with full redraws.
fn request_directional_references(
    settings: Res<ShadowCacheAuditSettings>,
    faults: Option<ResMut<ShadowCacheFaults>>,
    mut commands: Commands,
) {
    let wanted = settings.enabled && settings.fresh_redraws;
    match faults {
        Some(mut faults) => {
            if faults.directional_reference != wanted {
                faults.directional_reference = wanted;
            }
        }
        None if wanted => commands.insert_resource(ShadowCacheFaults {
            directional_reference: true,
            ..default()
        }),
        None => {}
    }
}

/// The lights the audit expects this frame, from the main world.
#[derive(Resource, Default)]
struct AuditPairs {
    settings: Option<ShadowCacheAuditSettings>,
    frame: u32,
    audit_frame: u32,
    /// (light, its render entity, its twin's render entity, if it has one):
    /// every shadowed spot light that is extracted (visible), twins aside.
    pairs: Vec<(Entity, Entity, Option<Entity>)>,
    /// (view, its light's render entity): every cached directional cascade
    /// the main world extracted.
    directional: Vec<(ShadowViewId, Entity)>,
    /// As `pairs`, for shadowed point lights (six views each).
    points: Vec<(Entity, Entity, Option<Entity>)>,
}

fn extract_audit_pairs(
    mut pairs: ResMut<AuditPairs>,
    settings: Extract<Res<ShadowCacheAuditSettings>>,
    frame: Extract<Option<Res<FrameCount>>>,
    lights: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &SpotLight,
                &ViewVisibility,
                Option<&HasShadowCacheAuditTwin>,
            ),
            (Without<ShadowCacheAuditTwin>, Without<ShadowCacheExempt>),
        >,
    >,
    twins: Extract<Query<RenderEntity, With<ShadowCacheAuditTwin>>>,
    directional_lights: Extract<Query<(Entity, RenderEntity, &DirectionalLightShadowCache)>>,
    point_lights: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &PointLight,
                &ViewVisibility,
                Option<&HasShadowCacheAuditTwin>,
            ),
            (Without<ShadowCacheAuditTwin>, Without<ShadowCacheExempt>),
        >,
    >,
) {
    pairs.settings = Some(**settings);
    pairs.frame = frame.as_deref().map_or(0, |frame| frame.0);
    pairs.audit_frame += 1;
    let mut expected: Vec<_> = lights
        .iter()
        .filter(|(_, _, light, visibility, _)| light.shadow_maps_enabled && visibility.get())
        .map(|(light, render_light, _, _, twin)| {
            (
                light,
                render_light,
                twin.and_then(|twin| twins.get(twin.0).ok()),
            )
        })
        .collect();
    expected.sort_unstable_by_key(|(light, ..)| *light);
    pairs.pairs = expected;
    let mut directional: Vec<_> = directional_lights
        .iter()
        .filter(|(_, _, cache)| cache.is_cached())
        .flat_map(|(light, render_light, cache)| {
            cache.views().map(move |(camera, _)| {
                (
                    ShadowViewId {
                        light,
                        subview: ShadowSubview::Directional {
                            camera,
                            cascade: cache.cascade(),
                        },
                    },
                    render_light,
                )
            })
        })
        .collect();
    directional.sort_unstable_by_key(|(view, _)| *view);
    pairs.directional = directional;
    let mut points: Vec<_> = point_lights
        .iter()
        .filter(|(_, _, light, visibility, _)| light.shadow_maps_enabled && visibility.get())
        .map(|(light, render_light, _, _, twin)| {
            (
                light,
                render_light,
                twin.and_then(|twin| twins.get(twin.0).ok()),
            )
        })
        .collect();
    points.sort_unstable_by_key(|(light, ..)| *light);
    pairs.points = points;
}

/// One pair's slice of the results buffer, in `u32` words; its first words
/// are the pair's parameters.
const RECORD_WORDS: usize = 128;
const RECORD_BYTES: u64 = (RECORD_WORDS * 4) as u64;
const MAX_PAIRS: usize = 64;
const MAX_TEXELS: usize = 16;

// Words of a record (see `COMPARE_SHADER`).
const WORKING_COUNT: usize = 4;
const STATIC_COUNT: usize = 6;
const WORKING_VS_STATIC_COUNT: usize = 8;
const WORKING_ULPS: usize = 10;
const STATIC_ULPS: usize = 11;
const TEXEL_COUNT: usize = 12;
const HASH_MODE: usize = 13;
const HASH_SUM: usize = 14;
const HASH_XOR: usize = 15;
const TEXELS: usize = 64;

const COMPARE_SHADER: &str = r"
@group(0) @binding(0) var maps: texture_depth_2d_array;
@group(0) @binding(1) var static_layer: texture_depth_2d;
// 0: the working layer, 1: the twin's layer, 2: 1 with a static layer, 3: 1 to
// list the texels where working != static instead of working != twin;
// 4, 5: working != twin (count, first texel as y << 16 | x); 6, 7: static !=
// twin; 8, 9: working != static; 10, 11: the largest difference in ulps of
// working and of static against the twin; 12: differing texels listed;
// 13: 1 to hash the working layer into 14 (a sum) and 15 (a xor);
// 64..: the listed texels (y << 16 | x, working, twin, static).
@group(0) @binding(2) var<storage, read_write> record: array<atomic<u32>, 128>;

fn ulps(a: u32, b: u32) -> u32 {
    return select(b - a, a - b, a > b);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(maps);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let packed = (id.y << 16u) | id.x;
    let w = bitcast<u32>(textureLoad(maps, texel, i32(atomicLoad(&record[0])), 0));
    if (atomicLoad(&record[13]) != 0u) {
        let place = (id.x * 73856093u) ^ (id.y * 19349663u) ^ 0x9e3779b9u;
        var a = (w ^ place) * 0x85ebca6bu;
        a = a ^ (a >> 13u);
        var b = (w + id.x * 2654435761u + id.y * 40503u) * 0xc2b2ae35u;
        b = b ^ (b >> 16u);
        atomicAdd(&record[14], a);
        atomicXor(&record[15], b);
        return;
    }
    let f = bitcast<u32>(textureLoad(maps, texel, i32(atomicLoad(&record[1])), 0));
    var s = 0u;
    if (atomicLoad(&record[2]) != 0u) {
        s = bitcast<u32>(textureLoad(static_layer, texel, 0));
        if (s != f) {
            atomicAdd(&record[6], 1u);
            atomicMin(&record[7], packed);
            atomicMax(&record[11], ulps(s, f));
        }
        if (w != s) {
            atomicAdd(&record[8], 1u);
            atomicMin(&record[9], packed);
        }
    }
    if (w != f) {
        atomicAdd(&record[4], 1u);
        atomicMin(&record[5], packed);
        atomicMax(&record[10], ulps(w, f));
    }
    let list_static = atomicLoad(&record[3]) != 0u;
    if ((!list_static && w != f) || (list_static && atomicLoad(&record[2]) != 0u && w != s)) {
        let slot = atomicAdd(&record[12], 1u);
        if (slot < 16u) {
            let base = 64u + slot * 4u;
            atomicStore(&record[base], packed);
            atomicStore(&record[base + 1u], w);
            atomicStore(&record[base + 2u], f);
            atomicStore(&record[base + 3u], s);
        }
    }
}
";

/// What a pair's compare reports besides the GPU's counts.
struct PendingPair {
    record: ShadowCacheAuditRecord,
    has_static: bool,
    hash: bool,
}

struct ReadbackSlot {
    records: Buffer,
    readback: Buffer,
    pending: Option<(Vec<PendingPair>, Arc<AtomicBool>)>,
}

struct AuditGpu {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
    /// Bound where a light has no static layer.
    no_static_layer: TextureView,
    slots: Vec<ReadbackSlot>,
}

impl AuditGpu {
    fn new(device: &RenderDevice) -> Self {
        let texture = |sample_type, view_dimension| BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        };
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(
            "shadow_cache_audit_layout",
            &[
                entry(
                    0,
                    texture(TextureSampleType::Depth, TextureViewDimension::D2Array),
                ),
                entry(
                    1,
                    texture(TextureSampleType::Depth, TextureViewDimension::D2),
                ),
                entry(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        );
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("shadow_cache_audit"),
            source: ShaderSource::Wgsl(Cow::Borrowed(COMPARE_SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_cache_audit"),
            bind_group_layouts: &[Some(&layout)],
            ..default()
        });
        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("shadow_cache_audit"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: default(),
            cache: None,
        });
        let no_static_layer = device
            .create_texture(&TextureDescriptor {
                label: Some("shadow_cache_audit_no_static_layer"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: CORE_3D_SHADOW_MAP_FORMAT,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());
        let buffer = |label, usage| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: MAX_PAIRS as u64 * RECORD_BYTES,
                usage,
                mapped_at_creation: false,
            })
        };
        let slots = (0..8)
            .map(|_| ReadbackSlot {
                records: buffer(
                    "shadow_cache_audit_records",
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                ),
                readback: buffer(
                    "shadow_cache_audit_readback",
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                ),
                pending: None,
            })
            .collect();
        Self {
            layout,
            pipeline,
            no_static_layer,
            slots,
        }
    }

    /// Moves the read-backs that arrived to `results`.
    fn drain(&mut self, results: &ShadowCacheAuditResults) {
        for slot in &mut self.slots {
            let Some((pairs, mapped)) = &mut slot.pending else {
                continue;
            };
            if !mapped.load(Ordering::Acquire) {
                continue;
            }
            {
                let data = slot.readback.slice(..).get_mapped_range().unwrap();
                let words: Vec<u32> = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| u32::from_le_bytes(*bytes))
                    .collect();
                let texel = |packed: u32| UVec2::new(packed & 0xffff, packed >> 16);
                let first = |count: u32, packed: u32| (count > 0).then(|| texel(packed));
                let Ok(mut out) = results.0.lock() else {
                    return;
                };
                for (index, pair) in pairs.drain(..).enumerate() {
                    let word = |at: usize| words[index * RECORD_WORDS + at];
                    let mut record = pair.record;
                    record.differing = word(WORKING_COUNT);
                    record.first = first(record.differing, word(WORKING_COUNT + 1));
                    record.max_ulps = word(WORKING_ULPS);
                    record.texels = (0..(word(TEXEL_COUNT) as usize).min(MAX_TEXELS))
                        .map(|listed| {
                            let at = TEXELS + listed * 4;
                            ShadowCacheAuditTexel {
                                texel: texel(word(at)),
                                working: word(at + 1),
                                fresh: word(at + 2),
                                static_layer: word(at + 3),
                            }
                        })
                        .collect();
                    if pair.hash {
                        record.hash = Some(((word(HASH_SUM) as u64) << 32) | word(HASH_XOR) as u64);
                    } else if pair.has_static {
                        let count = word(STATIC_COUNT);
                        record.static_differing = Some((
                            count,
                            first(count, word(STATIC_COUNT + 1)),
                            word(STATIC_ULPS),
                        ));
                        record.working_vs_static = Some(word(WORKING_VS_STATIC_COUNT));
                    }
                    out.push(record);
                }
            }
            slot.readback.unmap();
            slot.pending = None;
        }
    }
}

/// Where a light's (by render entity) shadow view of a kind (a spot light's,
/// a point light's face) drew this frame: its texture and layer.
fn drawn_target(
    lights: &Query<(
        Option<&ExtractedPointLight>,
        Option<&PointAndSpotLightViewEntities>,
    )>,
    keys: &Query<&ShadowViewKey>,
    light: Entity,
    subview: ShadowSubview,
) -> Option<(TextureId, u32)> {
    let Ok((Some(_), Some(views))) = lights.get(light) else {
        return None;
    };
    views
        .iter()
        .filter_map(|view| keys.get(*view).ok())
        .find(|key| key.id.subview == subview)
        .map(|key| (key.region.texture, key.region.layer))
}

/// After the frame's passes, before they are acknowledged: compares each
/// cached view's shadow map and static layer with its twin's full redraw.
///
/// Every expected view-frame gets a record: compared, skipped as dormant, or
/// missing with the reason. A comparison reads the texture the views'
/// lighting binds, and only after checking that the light and its twin drew
/// into that texture.
fn audit_shadow_caches(
    pairs: Res<AuditPairs>,
    caches: Res<ShadowViewCaches>,
    lights: Query<(
        Option<&ExtractedPointLight>,
        Option<&PointAndSpotLightViewEntities>,
    )>,
    keys: Query<&ShadowViewKey>,
    bindings: Query<&ViewShadowBindings>,
    cluster_meta: Res<GlobalClusterableObjectMeta>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    results: Res<ShadowCacheAuditResults>,
    mut gpu: Local<Option<AuditGpu>>,
) {
    let gpu = gpu.get_or_insert_with(|| AuditGpu::new(&device));
    let _ = device.poll(PollType::Poll);
    gpu.drain(&results);

    let Some(settings) = pairs.settings.filter(|settings| settings.enabled) else {
        return;
    };
    // The shadow map textures lighting reads: the same for every view.
    let mut bound = bindings.iter();
    let binding = match bound.next() {
        None => Err("no view binds a shadow map"),
        Some(first)
            if bound.any(|other| {
                other.directional_light_depth_texture.id()
                    != first.directional_light_depth_texture.id()
                    || other.point_light_depth_texture.id() != first.point_light_depth_texture.id()
            }) =>
        {
            Err("the views bind different shadow map textures")
        }
        Some(first) => Ok(first),
    };
    // The point light texture's faces as a 2D array, for the compare.
    let point_maps = binding.ok().map(|binding| {
        (
            binding
                .point_light_depth_texture
                .create_view(&TextureViewDescriptor {
                    label: Some("shadow_cache_audit_point_faces"),
                    dimension: Some(TextureViewDimension::D2Array),
                    aspect: TextureAspect::DepthOnly,
                    ..default()
                }),
            binding.point_light_depth_texture.width(),
        )
    });
    let hashes = settings.hashes && !settings.fresh_redraws;
    // (view, its light's render entity, the twin's): with twins or hashes
    // the view of every shadowed spot light the main world extracted, else
    // every cached view planned.
    let expected: Vec<(ShadowViewId, Entity, Option<Entity>)> = if settings.fresh_redraws || hashes
    {
        pairs
            .pairs
            .iter()
            .map(|&(light, render_light, twin)| (ShadowViewId::spot(light), render_light, twin))
            // Point lights: each face with the twin's.
            .chain(
                pairs
                    .points
                    .iter()
                    .flat_map(|&(light, render_light, twin)| {
                        (0..6u8).map(move |face| {
                            (
                                ShadowViewId {
                                    light,
                                    subview: ShadowSubview::PointFace(face),
                                },
                                render_light,
                                twin,
                            )
                        })
                    }),
            )
            // Cached directional cascades: compared with their references
            // (not hashed).
            .chain(
                pairs
                    .directional
                    .iter()
                    .filter(|_| settings.fresh_redraws)
                    .map(|&(view, render_light)| (view, render_light, None)),
            )
            .collect()
    } else {
        let mut cached: Vec<_> = caches
            .audited_views()
            .map(|(view, render_light)| (view, render_light, None))
            .collect();
        cached.sort_unstable_by_key(|(view, ..)| *view);
        cached
    };
    let count = expected.len() as u32;
    let mut audited = Vec::with_capacity(expected.len());
    let mut skipped = Vec::new();
    for (view, render_light, twin) in expected {
        let info = caches.audit(&view);
        let directional = matches!(view.subview, ShadowSubview::Directional { .. });
        let point = matches!(view.subview, ShadowSubview::PointFace(_));
        let mut record = ShadowCacheAuditRecord {
            frame: pairs.frame,
            expected: count,
            status: ShadowCacheAuditStatus::Compared,
            audit_frame: pairs.audit_frame,
            view,
            light: view.light,
            layers: (0, 0),
            fresh: settings.fresh_redraws,
            plan: info
                .as_ref()
                .map_or(ShadowCachePlan::Fallback, |info| info.plan),
            static_revision: info.as_ref().map_or(0, |info| info.static_revision),
            static_content: info.as_ref().and_then(|info| info.static_content),
            working: info.as_ref().and_then(|info| info.working),
            static_drawn: info.as_ref().is_some_and(|info| info.static_drawn),
            movers_texels: info.as_ref().map_or(0, |info| info.movers_texels),
            written_texels: info.as_ref().map_or(0, |info| info.written_texels),
            restored_texels: info.as_ref().map_or(0, |info| info.restored_texels),
            differing: 0,
            first: None,
            max_ulps: 0,
            texels: Vec::new(),
            static_differing: None,
            working_vs_static: None,
            hash: None,
            cached: info.is_some(),
        };
        // A light that isn't cached this frame (it changed and hasn't settled)
        // is a full redraw: compared with its twin all the same.
        let target = info
            .as_ref()
            .map(|info| (info.texture, info.layer))
            .or_else(|| drawn_target(&lights, &keys, render_light, view.subview));
        let twin_target = twin.and_then(|twin| drawn_target(&lights, &keys, twin, view.subview));
        // The layer lighting reads a point light's face from: its index among
        // the clustered lights (the cube array is indexed by it) times six,
        // plus the face; found apart from where the cache says it drew.
        let read_layer = |light: Entity| match view.subview {
            ShadowSubview::PointFace(face) => cluster_meta
                .entity_to_index
                .get(&light)
                .map(|index| (*index * 6 + face as usize) as u32),
            _ => None,
        };
        let read = binding.map(|binding| {
            if point {
                binding.point_light_depth_texture.id()
            } else {
                binding.directional_light_depth_texture.id()
            }
        });
        // A directional cascade's map and reference, copied and drawn into a
        // texture of the audit's during the camera's passes.
        let reference = directional
            .then(|| caches.reference(&view))
            .flatten()
            .filter(|_| settings.fresh_redraws);
        let status = if directional {
            match (&info, &reference) {
                (None, _) => ShadowCacheAuditStatus::Missing("the cascade wasn't planned"),
                (_, None) if settings.fresh_redraws => {
                    ShadowCacheAuditStatus::Missing("the cascade has no reference")
                }
                (_, Some((_, _, false))) => {
                    ShadowCacheAuditStatus::Missing("the reference view drew nothing")
                }
                _ => ShadowCacheAuditStatus::Compared,
            }
        } else if record.plan == ShadowCachePlan::Dormant {
            // It drew nothing: its map isn't current, and nothing reads it.
            ShadowCacheAuditStatus::Dormant
        } else {
            match (read, target, twin_target) {
                (Err(why), ..) => ShadowCacheAuditStatus::Missing(why),
                (_, None, _) => ShadowCacheAuditStatus::Missing("the light drew no shadow map"),
                (Ok(read), Some((texture, _)), _) if texture != read => {
                    ShadowCacheAuditStatus::Missing(
                        "the light drew into another texture than lighting reads",
                    )
                }
                (_, _, None) if settings.fresh_redraws => {
                    ShadowCacheAuditStatus::Missing(if twin.is_some() {
                        "the twin drew no shadow map"
                    } else {
                        "the light has no twin"
                    })
                }
                (Ok(read), _, Some((texture, _))) if settings.fresh_redraws && texture != read => {
                    ShadowCacheAuditStatus::Missing(
                        "the twin drew into another texture than lighting reads",
                    )
                }
                (_, Some((_, layer)), _) if point && read_layer(render_light) != Some(layer) => {
                    ShadowCacheAuditStatus::Missing(
                        "the light drew into another layer than lighting reads",
                    )
                }
                (_, _, Some((_, layer)))
                    if point
                        && settings.fresh_redraws
                        && read_layer(twin.unwrap()) != Some(layer) =>
                {
                    ShadowCacheAuditStatus::Missing(
                        "the twin drew into another layer than lighting reads",
                    )
                }
                _ => ShadowCacheAuditStatus::Compared,
            }
        };
        record.status = status;
        let layer = target.map_or(0, |(_, layer)| layer);
        record.layers = (
            layer,
            twin_target
                .filter(|_| settings.fresh_redraws)
                .map_or(layer, |(_, layer)| layer),
        );
        if reference.is_some() {
            // Layer 0 the cascade's map, layer 1 its reference.
            record.layers = (0, 1);
        }
        if status != ShadowCacheAuditStatus::Compared {
            skipped.push(record);
            continue;
        }
        let static_layer = info.and_then(|info| info.static_layer).filter(|_| !hashes);
        // The texture compared: a directional cascade's own (its map and
        // reference), the point light texture's faces, or the shadow map
        // texture of spot lights (`None`).
        let maps = match reference {
            Some((maps, size, _)) => Some((maps, size)),
            None if point => point_maps.clone(),
            None => None,
        };
        audited.push((
            PendingPair {
                has_static: static_layer.is_some(),
                hash: hashes && !directional,
                record,
            },
            static_layer,
            maps,
        ));
    }
    if !skipped.is_empty()
        && let Ok(mut records) = results.0.lock()
    {
        records.append(&mut skipped);
    }
    let Ok(binding) = binding else {
        return;
    };

    // In batches of the results buffer's size.
    while !audited.is_empty() {
        let rest = audited.split_off(audited.len().min(MAX_PAIRS));
        let batch = core::mem::replace(&mut audited, rest);
        if gpu.slots.iter().all(|slot| slot.pending.is_some()) {
            let _ = device.poll(PollType::wait_indefinitely());
            gpu.drain(&results);
        }
        gpu.submit(&device, &queue, binding, batch, &results);
    }
}

impl AuditGpu {
    /// Compares (or hashes) up to [`MAX_PAIRS`] lights in a free read-back
    /// slot; without one they are recorded as missing.
    fn submit(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        binding: &ViewShadowBindings,
        audited: Vec<(PendingPair, Option<TextureView>, Option<(TextureView, u32)>)>,
        results: &ShadowCacheAuditResults,
    ) {
        let Some(slot) = self.slots.iter_mut().find(|slot| slot.pending.is_none()) else {
            if let Ok(mut records) = results.0.lock() {
                records.extend(
                    audited
                        .into_iter()
                        .map(|(pair, ..)| ShadowCacheAuditRecord {
                            status: ShadowCacheAuditStatus::Missing("no free read-back slot"),
                            ..pair.record
                        }),
                );
            }
            return;
        };
        // Each record starts with its parameters, the counts at 0, the first
        // texels at the largest value.
        let mut init = vec![0u32; audited.len() * RECORD_WORDS];
        for (index, (pair, ..)) in audited.iter().enumerate() {
            let record = &mut init[index * RECORD_WORDS..][..RECORD_WORDS];
            record[0] = pair.record.layers.0;
            record[1] = pair.record.layers.1;
            record[2] = pair.has_static as u32;
            record[3] = !pair.record.fresh as u32;
            record[HASH_MODE] = pair.hash as u32;
            for first in [WORKING_COUNT, STATIC_COUNT, WORKING_VS_STATIC_COUNT] {
                record[first + 1] = u32::MAX;
            }
        }
        let init: Vec<u8> = init.into_iter().flat_map(u32::to_le_bytes).collect();
        queue.write_buffer(&slot.records, 0, &init);
        let size = binding.directional_light_depth_texture.width();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("shadow_cache_audit"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("shadow_cache_audit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (index, (_, static_layer, maps)) in audited.iter().enumerate() {
                // A directional cascade's own texture (its map and reference),
                // or the shadow map texture lighting reads.
                let (maps_view, size) = match maps {
                    Some((view, size)) => (view, *size),
                    None => (&binding.directional_light_depth_texture_view, size),
                };
                let bind_group = device.create_bind_group(
                    "shadow_cache_audit",
                    &self.layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(maps_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(
                                static_layer.as_ref().unwrap_or(&self.no_static_layer),
                            ),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &slot.records,
                                offset: index as u64 * RECORD_BYTES,
                                size: core::num::NonZero::new(RECORD_BYTES),
                            }),
                        },
                    ],
                );
                pass.set_bind_group(0, &*bind_group, &[]);
                pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);
            }
        }
        let bytes = audited.len() as u64 * RECORD_BYTES;
        encoder.copy_buffer_to_buffer(&slot.records, 0, &slot.readback, 0, bytes);
        queue.submit([encoder.finish()]);
        let mapped = Arc::new(AtomicBool::new(false));
        let flag = mapped.clone();
        slot.readback
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        slot.pending = Some((audited.into_iter().map(|(pair, ..)| pair).collect(), mapped));
    }
}
