//! Provides component types for lighting a bevy scene. This includes the usual
//! directional, point, and spot lights, as well as light probes, atmosphere,
//! other volumetrics, and shadow configuration.

extern crate alloc;

use bevy_app::{App, Plugin, PostUpdate, Update};
use bevy_asset::{AssetApp, AssetEventSystems};
use bevy_camera::{
    primitives::{Aabb, CascadesFrusta, CubemapFrusta, Frustum, Sphere},
    visibility::{
        CascadesVisibleEntities, CubemapVisibleEntities, DynamicSkinnedMeshBounds,
        InheritedVisibility, NoCpuCulling, NoFrustumCulling, RenderLayers, ViewVisibility,
        VisibilityRange, VisibilitySystems, VisibleEntities, VisibleEntityRanges,
        VisibleMeshEntities,
    },
    Camera, Camera3d, CameraUpdateSystems, RenderTarget, ShadowLodOrigin,
};
use bevy_ecs::{
    entity::{EntityHashMap, EntityHashSet},
    prelude::*,
    system::QueryLens,
};
#[cfg(feature = "bevy_gizmos")]
use bevy_gizmos::frustum::FrustumGizmoSystems;
use bevy_log::warn_once;
use bevy_math::Vec3A;
use bevy_mesh::{
    morph::{MeshMorphWeights, MorphWeights},
    skinning::SkinnedMesh,
    Mesh3d,
};
use bevy_reflect::prelude::*;
use bevy_transform::{components::GlobalTransform, TransformSystems};
use bevy_utils::Parallel;
use core::{any::TypeId, mem, ops::DerefMut};
use smallvec::{smallvec, SmallVec};

pub mod cluster;
use cluster::assign::assign_objects_to_clusters;
pub use cluster::ClusteredDecal;
mod ambient_light;
pub use ambient_light::{AmbientLight, GlobalAmbientLight};
use bevy_camera::visibility::SetViewVisibility;

mod probe;
pub use probe::{
    automatically_add_parallax_correction_components, AtmosphereEnvironmentMapLight,
    EnvironmentMapLight, EnvironmentMapSettings, GeneratedEnvironmentMapLight,
    GeneratedEnvironmentMapSpecularOnly, IrradianceVolume, LightProbe, ParallaxCorrection,
    ReflectionCaptureOrigin, ReflectionProbePriority, Skybox,
};
pub mod atmosphere;
pub use atmosphere::Atmosphere;
mod volumetric;
pub use volumetric::{FogVolume, VolumetricFog, VolumetricLight};
pub mod cascade;
use cascade::build_directional_light_cascades;
mod stable_cascade;
pub use cascade::{CascadeShadowConfig, CascadeShadowConfigBuilder, Cascades};
use stable_cascade::view_takes;
pub use stable_cascade::{
    update_shadow_scene_bounds, ShadowRepresentation, ShadowSceneBounds, StableCascadeProjections,
    StableCascades,
};
mod point_light;
pub use point_light::{
    update_point_light_frusta, PointLight, PointLightShadowMap, PointLightTexture,
};
mod spot_light;
pub use spot_light::{
    orthonormalize, spot_light_clip_from_view, spot_light_world_from_view,
    update_spot_light_frusta, SpotLight, SpotLightTexture,
};
mod directional_light;
pub use directional_light::{
    update_directional_light_frusta, DirectionalLight, DirectionalLightShadowMap,
    DirectionalLightTexture, SunDisk,
};
mod rect_light;
pub use rect_light::RectLight;
mod shadow_cache;
pub use shadow_cache::{
    classify_shadow_casters, invalidate_shadow_caches, DirectionalLightShadowCache,
    InvalidateShadowCache, LiveShadowMaterial, LiveShadowMaterials, ShadowCacheCounters,
    ShadowCacheExempt, ShadowCacheFaults, ShadowCacheFeedback, ShadowCacheLoading,
    ShadowCacheSettings, ShadowCacheState, ShadowCacheViewStatus, ShadowCasterChange,
    ShadowCasterLive, ShadowCasterState, ShadowFootprint, ShadowSubview, ShadowTileMask,
    ShadowViewCache, ShadowViewFrameCounts, ShadowViewId, SpotLightShadowCache, StaticShadowCaster,
};
use shadow_cache::{
    DirectionalLightSettle, DirectionalShadowProjection, MoversGrid, ShadowViewProjection,
    SpotShadowProjection,
};

/// Provides gizmo drawing for visualizing light positions.
#[cfg(feature = "bevy_gizmos")]
pub mod gizmos;

/// The light prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        light_consts, AmbientLight, DirectionalLight, EnvironmentMapLight,
        GeneratedEnvironmentMapLight, GlobalAmbientLight, LightProbe, PointLight, RectLight,
        SpotLight,
    };

    #[doc(hidden)]
    #[cfg(feature = "bevy_gizmos")]
    pub use crate::gizmos::{LightGizmoColor, LightGizmoConfigGroup, ShowLightGizmo};
}

use crate::{
    atmosphere::{extract_chromatic_phase_textures, ScatteringMedium},
    cluster::{add_light_probe_and_decal_aabbs, ClusterVisibilityClass, Clusters},
    directional_light::validate_shadow_map_size,
    point_light::update_point_light_bounding_spheres,
    spot_light::update_spot_light_bounding_spheres,
};

/// Constants for operating with the light units: lumens, and lux.
pub mod light_consts {
    /// Approximations for converting the wattage of lamps to lumens.
    ///
    /// The **lumen** (symbol: **lm**) is the unit of [luminous flux], a measure
    /// of the total quantity of [visible light] emitted by a source per unit of
    /// time, in the [International System of Units] (SI).
    ///
    /// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lumen_(unit))
    ///
    /// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
    /// [visible light]: https://en.wikipedia.org/wiki/Visible_light
    /// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
    pub mod lumens {
        /// The conversion factor used to determine how many lumens a typical LED light of a given wattage produces.
        pub const LUMENS_PER_LED_WATTS: f32 = 90.0;
        /// The conversion factor used to determine how many lumens a typical incandescent light of a given wattage produces.
        pub const LUMENS_PER_INCANDESCENT_WATTS: f32 = 13.8;
        /// The conversion factor used to determine how many lumens a typical halogen light of a given wattage produces.
        pub const LUMENS_PER_HALOGEN_WATTS: f32 = 19.8;
        /// 1,000,000 lumens is a very large "cinema light" capable of registering brightly at Bevy's
        /// default [`bevy_camera::Exposure::BLENDER`] exposure level. For "indoor lighting" with a lower exposure,
        /// this would be way too bright.
        pub const VERY_LARGE_CINEMA_LIGHT: f32 = 1_000_000.0;
    }

    /// Predefined for lux values in several locations.
    ///
    /// The **lux** (symbol: **lx**) is the unit of [illuminance], or [luminous flux] per unit area,
    /// in the [International System of Units] (SI). It is equal to one lumen per square meter.
    ///
    /// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lux)
    ///
    /// [illuminance]: https://en.wikipedia.org/wiki/Illuminance
    /// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
    /// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
    pub mod lux {
        /// The amount of light (lux) in a moonless, overcast night sky. (starlight)
        pub const MOONLESS_NIGHT: f32 = 0.0001;
        /// The amount of light (lux) during a full moon on a clear night.
        pub const FULL_MOON_NIGHT: f32 = 0.05;
        /// The amount of light (lux) during the dark limit of civil twilight under a clear sky.
        pub const CIVIL_TWILIGHT: f32 = 3.4;
        /// The amount of light (lux) in family living room lights.
        pub const LIVING_ROOM: f32 = 50.;
        /// The amount of light (lux) in an office building's hallway/toilet lighting.
        pub const HALLWAY: f32 = 80.;
        /// The amount of light (lux) in very dark overcast day
        pub const DARK_OVERCAST_DAY: f32 = 100.;
        /// The amount of light (lux) in an office.
        pub const OFFICE: f32 = 320.;
        /// The amount of light (lux) during sunrise or sunset on a clear day.
        pub const CLEAR_SUNRISE: f32 = 400.;
        /// The amount of light (lux) on an overcast day; typical TV studio lighting
        pub const OVERCAST_DAY: f32 = 1000.;
        /// The amount of light (lux) from ambient daylight (not direct sunlight).
        /// This is the default for [`DirectionalLight`](crate::DirectionalLight)s in Bevy.
        pub const AMBIENT_DAYLIGHT: f32 = 10_000.;
        /// The amount of light (lux) in full daylight (not direct sun).
        pub const FULL_DAYLIGHT: f32 = 20_000.;
        /// The amount of light (lux) in direct sunlight.
        pub const DIRECT_SUNLIGHT: f32 = 100_000.;
        /// The amount of light (lux) of raw sunlight, not filtered by the atmosphere.
        pub const RAW_SUNLIGHT: f32 = 130_000.;
    }
}

/// Sets up all the light visibility and clustering infrastructure needed for rendering lights.
#[derive(Default)]
pub struct LightPlugin;

impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlobalAmbientLight>()
            .init_resource::<DirectionalLightShadowMap>()
            .init_resource::<PointLightShadowMap>()
            .init_resource::<ShadowCacheSettings>()
            .init_resource::<ShadowCacheFeedback>()
            .init_resource::<ShadowSceneBounds>()
            .add_message::<InvalidateShadowCache>()
            .init_asset::<ScatteringMedium>()
            .register_required_components::<Camera3d, Clusters>()
            .register_required_components::<SpotLight, SpotLightShadowCache>()
            .register_required_components::<DirectionalLight, DirectionalLightShadowCache>()
            .configure_sets(
                PostUpdate,
                SimulationLightSystems::CheckLightVisibility
                    .ambiguous_with(SimulationLightSystems::CheckLightVisibility),
            )
            .configure_sets(
                PostUpdate,
                SimulationLightSystems::AssignLightsToClusters
                    .before(bevy_app::TransformGizmoRenderStep),
            )
            .add_systems(Update, automatically_add_parallax_correction_components)
            .add_systems(
                PostUpdate,
                (
                    validate_shadow_map_size.before(build_directional_light_cascades),
                    assign_objects_to_clusters
                        .in_set(SimulationLightSystems::AssignLightsToClusters)
                        .after(TransformSystems::Propagate)
                        .after(VisibilitySystems::CheckVisibility)
                        .after(CameraUpdateSystems),
                    update_directional_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        // This must run after CheckVisibility because it relies on `ViewVisibility`
                        .after(VisibilitySystems::CheckVisibility)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::UpdateDirectionalLightCascades)
                        // We assume that no entity will be both a directional light and a spot light,
                        // so these systems will run independently of one another.
                        // FIXME: Add an archetype invariant for this https://github.com/bevyengine/bevy/issues/1481.
                        .ambiguous_with(update_spot_light_frusta),
                    update_point_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    #[cfg(feature = "bevy_gizmos")]
                    update_spot_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .before(FrustumGizmoSystems)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    #[cfg(not(feature = "bevy_gizmos"))]
                    update_spot_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    (
                        check_dir_light_mesh_visibility.after(invalidate_shadow_caches),
                        // MPS: the casters' settled state, then the cached
                        // lights it redraws, then the lights' lists, all in
                        // one frame (the markers' commands are applied
                        // between them).
                        classify_shadow_casters
                            .after(AssetEventSystems)
                            .before(invalidate_shadow_caches),
                        check_point_light_mesh_visibility.after(invalidate_shadow_caches),
                        invalidate_shadow_caches,
                    )
                        .in_set(SimulationLightSystems::CheckLightVisibility)
                        .after(VisibilitySystems::CalculateBounds)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::UpdateLightFrusta)
                        // Lights can "see" entities and mark them as visible. This is done to
                        // correctly render shadows for entities that are not in view of a camera,
                        // but must be renderable to cast shadows. Because of this, we need to check
                        // entity visibility and mark as visible before they can be hidden.
                        .after(VisibilitySystems::CheckVisibility)
                        .before(VisibilitySystems::MarkNewlyHiddenEntitiesInvisible),
                    (
                        update_point_light_bounding_spheres.after(TransformSystems::Propagate),
                        update_spot_light_bounding_spheres.after(TransformSystems::Propagate),
                        add_light_probe_and_decal_aabbs,
                    )
                        .in_set(SimulationLightSystems::UpdateBounds)
                        .before(VisibilitySystems::UpdateFrusta),
                    // MPS: the scene's bounds, for the depth of stable far
                    // cascades, before the cascades are built.
                    update_shadow_scene_bounds
                        .run_if(any_with_component::<StableCascades>)
                        .after(VisibilitySystems::CalculateBounds)
                        .after(TransformSystems::Propagate)
                        .before(SimulationLightSystems::UpdateDirectionalLightCascades),
                    build_directional_light_cascades
                        .in_set(SimulationLightSystems::UpdateDirectionalLightCascades)
                        .after(TransformSystems::Propagate)
                        .before(bevy_app::TransformGizmoRenderStep)
                        .after(CameraUpdateSystems),
                    extract_chromatic_phase_textures.after(AssetEventSystems),
                ),
            );

        shadow_cache::add_shadow_caster_tracking(app);

        #[cfg(feature = "bevy_gizmos")]
        app.add_plugins(gizmos::LightGizmoPlugin);
    }
}

/// A convenient alias for `Or<(With<PointLight>, With<SpotLight>,
/// With<DirectionalLight>, With<RectLight>)>`, for use with [`bevy_camera::visibility::VisibleEntities`].
pub type WithLight = Or<(
    With<PointLight>,
    With<SpotLight>,
    With<DirectionalLight>,
    With<RectLight>,
)>;

/// Add this component to make a [`Mesh3d`] not cast shadows.
#[derive(Debug, Component, Reflect, Default, Clone, PartialEq)]
#[reflect(Component, Default, Debug, Clone, PartialEq)]
pub struct NotShadowCaster;

/// Add this component to make a [`Mesh3d`] not receive shadows.
///
/// **Note:** If you're using diffuse transmission, setting [`NotShadowReceiver`] will
/// cause both “regular” shadows as well as diffusely transmitted shadows to be disabled,
/// even when [`TransmittedShadowReceiver`] is being used.
#[derive(Debug, Component, Reflect, Default, Clone)]
#[reflect(Component, Default, Debug)]
pub struct NotShadowReceiver;
/// Add this component to make a [`Mesh3d`] using a PBR material with `StandardMaterial::diffuse_transmission > 0.0`
/// receive shadows on its diffuse transmission lobe. (i.e. its “backside”)
///
/// Not enabled by default, as it requires carefully setting up `StandardMaterial::thickness`
/// (and potentially even baking a thickness texture!) to match the geometry of the mesh, in order to avoid self-shadow artifacts.
///
/// **Note:** Using [`NotShadowReceiver`] overrides this component.
#[derive(Debug, Component, Reflect, Default, Clone)]
#[reflect(Component, Default, Debug)]
pub struct TransmittedShadowReceiver;

/// Add this component to a [`Camera3d`]
/// to control how to anti-alias shadow edges.
///
/// The different modes use different approaches to
/// [Percentage Closer Filtering](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing).
#[derive(Debug, Component, Reflect, Clone, Copy, PartialEq, Eq, Default)]
#[reflect(Component, Default, Debug, PartialEq, Clone)]
pub enum ShadowFilteringMethod {
    /// Hardware 2x2.
    ///
    /// Fast but poor quality.
    Hardware2x2,
    /// Approximates a fixed Gaussian blur, good when TAA isn't in use.
    ///
    /// Good quality, good performance.
    ///
    /// For directional and spot lights, this uses a [method by Ignacio Castaño
    /// for *The Witness*] using 9 samples and smart filtering to achieve the same
    /// as a regular 5x5 filter kernel.
    ///
    /// [method by Ignacio Castaño for *The Witness*]: https://web.archive.org/web/20230210095515/http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/
    #[default]
    Gaussian,
    /// A randomized filter that varies over time, good when TAA is in use.
    ///
    /// Good quality when used with `TemporalAntiAliasing`
    /// and good performance.
    ///
    /// For directional and spot lights, this uses a [method by Jorge Jimenez for
    /// *Call of Duty: Advanced Warfare*] using 8 samples in spiral pattern,
    /// randomly-rotated by interleaved gradient noise with spatial variation.
    ///
    /// [method by Jorge Jimenez for *Call of Duty: Advanced Warfare*]: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
    Temporal,
}

/// System sets used to run light-related systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum SimulationLightSystems {
    /// The set that adds AABBs and bounding spheres to clustered objects.
    UpdateBounds,
    /// After this set, all lights have been clustered.
    AssignLightsToClusters,
    /// After this set, all directional light cascades are up to date.
    UpdateDirectionalLightCascades,
    /// After this set, the frusta of shadow-casting point lights, spot lights, and directional lights are up to date.
    UpdateLightFrusta,
    /// System order ambiguities between systems in this set are ignored:
    /// the order of systems within this set is irrelevant, as the various visibility-checking systems
    /// assumes that their operations are irreversible during the frame.
    CheckLightVisibility,
}

/// Updates the visibility for [`DirectionalLight`]s so that shadow map
/// rendering can work.
///
/// This only processes entities without [`NoCpuCulling`]. Entities with
/// [`NoCpuCulling`] receive no view-specific processing in the main world.
///
/// MPS: a cached light ([`DirectionalLightShadowCache::is_cached`]) caches the
/// far cascade of each camera: that cascade's [`VisibleMeshEntities`] list
/// only its live casters, every frame, with the tiles of its shadow map their
/// [`ShadowFootprint`]s touch; its settled casters ([`StaticShadowCaster`])
/// go to the view's [`ShadowViewCache`], and are visited only when the render
/// world may draw them (as for a cached spot light). Each cascade takes
/// casters by their [`ShadowRepresentation`].
pub fn check_dir_light_mesh_visibility(
    mut commands: Commands,
    mut directional_lights: Query<
        (
            Entity,
            (&DirectionalLight, &GlobalTransform, &Cascades),
            &CascadesFrusta,
            &mut CascadesVisibleEntities,
            Option<&RenderLayers>,
            &ViewVisibility,
            Option<&StableCascades>,
            (
                Option<Mut<DirectionalLightShadowCache>>,
                Has<ShadowCacheExempt>,
                Option<&StableCascadeProjections>,
            ),
        ),
        Without<SpotLight>,
    >,
    visible_entity_query: Query<
        (
            Entity,
            &InheritedVisibility,
            Option<&RenderLayers>,
            Option<Ref<Aabb>>,
            Option<&GlobalTransform>,
            Has<VisibilityRange>,
            Has<NoFrustumCulling>,
            Option<&ShadowRepresentation>,
            Has<StaticShadowCaster>,
            (
                Has<SkinnedMesh>,
                Has<DynamicSkinnedMeshBounds>,
                Has<MeshMorphWeights>,
                Has<MorphWeights>,
                Option<&LiveShadowMaterial>,
            ),
        ),
        (
            Without<NotShadowCaster>,
            Without<DirectionalLight>,
            Without<NoCpuCulling>,
            With<Mesh3d>,
        ),
    >,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    (shadow_map, shadow_cache_feedback, shadow_cache_settings, time, faults): (
        Res<DirectionalLightShadowMap>,
        Res<ShadowCacheFeedback>,
        Res<ShadowCacheSettings>,
        Option<Res<bevy_time::Time>>,
        Option<Res<ShadowCacheFaults>>,
    ),
    mut defer_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
    mut view_visible_entities_queue: Local<Parallel<Vec<Vec<Entity>>>>,
    (mut far_queue, mut frame, mut far_scratch): (
        Local<Parallel<Vec<FarCascadeQueue>>>,
        Local<u32>,
        Local<Vec<Entity>>,
    ),
) {
    let visible_entity_ranges = visible_entity_ranges.as_deref();
    *frame = frame.wrapping_add(1);
    let frame = *frame;
    let now = time.map_or(core::time::Duration::ZERO, |time| time.elapsed());
    let stale_ack = faults.as_ref().is_some_and(|faults| faults.stale_ack);
    let reference = faults
        .as_ref()
        .is_some_and(|faults| faults.directional_reference);
    let stale_projection = faults
        .as_ref()
        .is_some_and(|faults| faults.stale_projection);
    let wrong_cascade = faults.as_ref().is_some_and(|faults| faults.wrong_cascade);
    let no_stagger = faults.as_ref().is_some_and(|faults| faults.no_stagger);
    let early_view_settle = faults
        .as_ref()
        .is_some_and(|faults| faults.early_view_settle);
    let middle_stand_in_routing = faults.is_some_and(|faults| faults.middle_stand_in_routing);
    let map_size = shadow_map.size as u32;
    let mut shadow_cache_status = None;
    // MPS: the stable views drawn whole while they settle, for their frame
    // counts.
    let mut settling: SmallVec<[ShadowViewId; 4]> = SmallVec::new();
    let mut stock: SmallVec<[ShadowViewId; 4]> = SmallVec::new();

    for (
        light_entity,
        (directional_light, light_transform, cascades),
        frusta,
        mut visible_entities,
        maybe_view_mask,
        light_view_visibility,
        stable_far,
        (mut light_cache, exempt, stable_projections),
    ) in &mut directional_lights
    {
        let mut views_to_remove = Vec::new();
        for (view, cascade_view_entities) in &mut visible_entities.entities {
            match frusta.frusta.get(view) {
                Some(view_frusta) => {
                    cascade_view_entities.resize(view_frusta.len(), Default::default());
                }
                None => views_to_remove.push(*view),
            };
        }
        for (view, frusta) in &frusta.frusta {
            visible_entities
                .entities
                .entry(*view)
                .or_insert_with(|| vec![VisibleMeshEntities::default(); frusta.len()]);
        }

        for v in views_to_remove {
            visible_entities.entities.remove(&v);
        }

        let view_mask = maybe_view_mask.unwrap_or_default();
        if let Some(cache) = light_cache.as_mut() {
            let cache = cache.bypass_change_detection();
            cache.entered = 0;
            cache.longest_wait = 0;
        }

        // MPS: whether the light is cached: the cache on, shadows on, stable
        // cascades, and what projects its maps settled. A light
        // that turns is drawn whole every frame until it settles, with no
        // cache work.
        let cached = light_cache.as_mut().is_some_and(|cache| {
            let identity = DirectionalLightSettle {
                rotation: light_transform.rotation().to_array().map(f32::to_bits),
                map_size: shadow_map.size,
                render_layers: view_mask.clone(),
            };
            let settled = cache.bypass_change_detection().settlement.settle(
                &identity,
                frame,
                now,
                shadow_cache_settings.settle_time,
            );
            shadow_cache_settings.enabled
                && directional_light.shadow_maps_enabled
                && light_view_visibility.get()
                && !exempt
                && stable_far.is_some()
                && settled
        });
        if let Some(cache) = light_cache.as_mut()
            && (cache.cached != cached || (!cached && !cache.views.is_empty()))
        {
            cache.cached = cached;
            // Drawn again from scratch once it is cached again.
            cache.views.clear();
        }

        // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
        if !directional_light.shadow_maps_enabled || !light_view_visibility.get() {
            visible_entities.entities.clear();
            continue;
        }

        if let Some(cache) = light_cache.as_mut().filter(|_| cached) {
            let cache = cache.bypass_change_detection();
            cache.views.retain(|(camera, cascade), _| {
                frusta.frusta.get(camera).is_some_and(|view_frusta| {
                    stable_far.is_some_and(|policy| {
                        policy.is_stable(*cascade as usize, view_frusta.len())
                    })
                })
            });

            // MPS: every stable view settles on its projection; a settled
            // view that isn't cached waits its turn: at most one starts being
            // cached a frame, the one that waited longest, so no view waits
            // for ever behind others that re-center often.
            let policy = stable_far.unwrap();
            let mut candidates: SmallVec<[((Entity, u32), u32); 4]> = SmallVec::new();
            let mut waiting: SmallVec<[(Entity, u32); 4]> = SmallVec::new();
            for (view, view_frusta) in &frusta.frusta {
                let cascade_count = view_frusta.len();
                for cascade in policy.stable(cascade_count) {
                    let Some(stable_cascade) = cascades
                        .cascades
                        .get(view)
                        .and_then(|view_cascades| view_cascades.get(cascade))
                    else {
                        continue;
                    };
                    let projection = DirectionalShadowProjection::new(
                        &stable_cascade.clip_from_world,
                        shadow_map.size,
                        view_mask,
                        cascade_selects(policy, cascade, cascade_count, middle_stand_in_routing),
                    );
                    let key = (*view, cascade as u32);
                    let state = cache.views.entry(key).or_default();
                    // Bevy's projection for now (the speed fallback): drawn
                    // whole, at stock cost, no cache work.
                    if stable_projections
                        .is_some_and(|projections| projections.is_stock(key.0, key.1))
                    {
                        *state = Default::default();
                        stock.push(ShadowViewId {
                            light: light_entity,
                            subview: ShadowSubview::Directional {
                                camera: key.0,
                                cascade: key.1,
                            },
                        });
                        continue;
                    }
                    let settled = state.settlement.settle(
                        &projection,
                        frame,
                        now,
                        shadow_cache_settings.settle_time,
                    );
                    // (The faults keep a view whose projection changed: the
                    // second also keeps its static layer as it was.)
                    if !(settled || early_view_settle || stale_projection) {
                        state.view = None;
                        state.waiting_since = None;
                        waiting.push(key);
                        continue;
                    }
                    if state.view.is_some() {
                        continue;
                    }
                    if no_stagger {
                        state.view = Some(ShadowViewCache::default());
                        cache.entered += 1;
                        continue;
                    }
                    let since = *state.waiting_since.get_or_insert(frame);
                    waiting.push(key);
                    candidates.push((key, since));
                }
            }
            if let Some(key) = shadow_cache::longest_waiting(candidates)
                && let Some(state) = cache.views.get_mut(&key)
            {
                state.view = Some(ShadowViewCache::default());
                state.waiting_since = None;
                cache.entered += 1;
            }
            for key in waiting {
                let Some(state) = cache.views.get(&key) else {
                    continue;
                };
                if state.view.is_none() {
                    cache.longest_wait = cache.longest_wait.max(
                        state
                            .waiting_since
                            .map_or(0, |since| frame.wrapping_sub(since)),
                    );
                    settling.push(ShadowViewId {
                        light: light_entity,
                        subview: ShadowSubview::Directional {
                            camera: key.0,
                            cascade: key.1,
                        },
                    });
                }
            }
        }

        for (view, view_frusta) in &frusta.frusta {
            let cascade_count = view_frusta.len();
            // MPS: the stand-ins each cascade draws instead of the detail
            // (`ShadowRepresentation`), whether or not it is cached. (The
            // fault: the middle cascade takes the far one's.)
            let selects: SmallVec<[Option<u32>; 4]> = (0..cascade_count)
                .map(|index| {
                    stable_far.and_then(|policy| {
                        cascade_selects(policy, index, cascade_count, middle_stand_in_routing)
                    })
                })
                .collect();

            // MPS: the cached stable cascades of this camera (admitted above),
            // and how each is walked.
            let mut walks: SmallVec<[FarWalk; 2]> = SmallVec::new();
            if let (Some(cache), Some(policy)) = (light_cache.as_mut(), stable_far)
                && cached
            {
                let cache = cache.bypass_change_detection();
                for cascade in policy.stable(cascade_count) {
                    let Some(stable_cascade) = cascades
                        .cascades
                        .get(view)
                        .and_then(|view_cascades| view_cascades.get(cascade))
                    else {
                        continue;
                    };
                    let Some(view_cache) = cache
                        .views
                        .get_mut(&(*view, cascade as u32))
                        .and_then(|state| state.view.as_mut())
                    else {
                        continue;
                    };
                    let projection = DirectionalShadowProjection::new(
                        &stable_cascade.clip_from_world,
                        shadow_map.size,
                        view_mask,
                        selects[cascade],
                    );
                    let id = ShadowViewId {
                        light: light_entity,
                        subview: ShadowSubview::Directional {
                            camera: *view,
                            cascade: cascade as u32,
                        },
                    };
                    view_cache.begin();
                    if !view_cache.projection.as_ref().is_some_and(|cached| {
                        cached.is_directional(&projection) || stale_projection
                    }) {
                        view_cache.invalidated = true;
                    }
                    let invalidated = mem::take(&mut view_cache.invalidated);
                    if invalidated {
                        view_cache.static_revision += 1;
                        view_cache.projection = Some(ShadowViewProjection::Directional(projection));
                    }
                    view_cache.lights_a_view = true;
                    let statuses =
                        shadow_cache_status.get_or_insert_with(|| shadow_cache_feedback.views());
                    let status = view_cache.status(statuses, &id);
                    view_cache.single_view = status.is_some_and(|status| status.fallback);
                    let walk_statics = invalidated
                        || status.is_none_or(|status| {
                            let acknowledged = status.acknowledged_static_revision;
                            acknowledged != Some(view_cache.static_revision)
                                && !(stale_ack
                                    && acknowledged == Some(view_cache.static_revision - 1))
                        });
                    walks.push(FarWalk {
                        cascade,
                        clip_from_world: cascades.cascades[view]
                            [cascade - usize::from(wrong_cascade)]
                        .clip_from_world,
                        single_view: view_cache.single_view,
                        walk_statics,
                        invalidated,
                    });
                }
            }

            // Each cascade's walk, if it is cached.
            let mut walk_slots: SmallVec<[Option<usize>; 4]> = smallvec![None; cascade_count];
            for (slot, walk) in walks.iter().enumerate() {
                walk_slots[walk.cascade] = Some(slot);
            }

            // Resize any per-thread buffer left at a stale state from a prior frame.
            // Threads receiving no work this frame won't run the `init` closure from `par_iter`, and without this
            // their buffer might be indexed out-of-bounds during collection if the number of cascades has increased.
            for thread_queue in view_visible_entities_queue.iter_mut() {
                thread_queue.resize(cascade_count, Vec::default());
            }

            visible_entity_query.par_iter().for_each_init(
                || {
                    let mut entities = view_visible_entities_queue.borrow_local_mut();
                    entities.resize(cascade_count, Vec::default());
                    (
                        defer_visible_entities_queue.borrow_local_mut(),
                        entities,
                        far_queue.borrow_local_mut(),
                    )
                },
                |(
                    defer_visible_entities_local_queue,
                    view_visible_entities_local_queue,
                    far_locals,
                ),
                 (
                    entity,
                    inherited_visibility,
                    maybe_entity_mask,
                    maybe_aabb,
                    maybe_transform,
                    has_visibility_range,
                    has_no_frustum_culling,
                    representation,
                    is_static,
                    (skinned, dynamic_skinned_bounds, mesh_morphed, morphed, live_material),
                )| {
                    if !inherited_visibility.get() {
                        return;
                    }

                    let entity_mask = maybe_entity_mask.unwrap_or_default();
                    if !view_mask.intersects(entity_mask) {
                        return;
                    }

                    // Check visibility ranges.
                    if has_visibility_range
                        && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                            !visible_entity_ranges.entity_is_in_range_of_view(entity, *view)
                        })
                    {
                        return;
                    }

                    let representation = representation.copied();
                    let mut visible = false;
                    if far_locals.len() < walks.len() {
                        far_locals.resize_with(walks.len(), Default::default);
                    }
                    for (index, frustum_visible_entities) in
                        view_visible_entities_local_queue.iter_mut().enumerate()
                    {
                        if !view_takes(selects[index], representation) {
                            continue;
                        }
                        let slot = walk_slots[index];
                        let far = slot.map(|slot| &walks[slot]);
                        // MPS: a settled caster of a cached far cascade is
                        // visited only while its static content is drawn (or
                        // for the audit's reference list).
                        let split_static = far.is_some_and(|far| !far.single_view) && is_static;
                        let skip_static = split_static && !far.is_some_and(|far| far.walk_statics);
                        if skip_static && !(reference && far.is_some()) {
                            continue;
                        }
                        // Disable near-plane culling, as a shadow caster could lie before the near plane.
                        if let (Some(aabb), Some(transform)) = (&maybe_aabb, maybe_transform)
                            && !has_no_frustum_culling
                            && !view_frusta[index].intersects_obb(
                                aabb,
                                &transform.affine(),
                                false,
                                true,
                            )
                        {
                            continue;
                        }
                        visible = true;
                        let (Some(far), Some(slot)) = (far, slot) else {
                            frustum_visible_entities.push(entity);
                            continue;
                        };
                        let far_local = &mut far_locals[slot];
                        if !far_local.used {
                            far_local.used = true;
                            far_local.statics.clear();
                            far_local.reference.clear();
                            far_local.tiles.reset(map_size);
                            far_local.print = StaticCastersPrint::default();
                            far_local.live = 0;
                        }
                        if reference {
                            far_local.reference.push(entity);
                        }
                        if skip_static {
                            continue;
                        }
                        if split_static {
                            far_local.statics.push(entity);
                            continue;
                        }
                        frustum_visible_entities.push(entity);
                        if is_static {
                            // (Only a single view's list holds settled
                            // casters.)
                            far_local.print.add(entity);
                            continue;
                        }
                        far_local.live += 1;
                        if far.single_view {
                            continue;
                        }
                        // Where the live caster falls in the map: bounds that
                        // may not hold the mesh this frame (skinned without
                        // updated bounds, morphed) or none cover it whole.
                        let footprint = match (&maybe_aabb, maybe_transform) {
                            (Some(aabb), Some(transform))
                                if !has_no_frustum_culling
                                    && !(mesh_morphed || morphed)
                                    && (!skinned
                                        || (dynamic_skinned_bounds && aabb.is_changed())) =>
                            {
                                let world_from_local = transform.affine();
                                padded_aabb(aabb, &world_from_local, live_material).map_or(
                                    ShadowFootprint::Whole,
                                    |aabb| {
                                        ShadowFootprint::of_box_orthographic(
                                            &far.clip_from_world,
                                            &aabb,
                                            &world_from_local,
                                        )
                                    },
                                )
                            }
                            _ => ShadowFootprint::Whole,
                        };
                        far_local.tiles.insert(footprint);
                    }
                    if visible {
                        defer_visible_entities_local_queue.push(entity);
                    }
                },
            );
            // collect entities from parallel queue
            for (view_dest_index, view_dest) in visible_entities
                .entities
                .get_mut(view)
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                view_dest.entities.clear();
                for thread_entity_queue in view_visible_entities_queue.iter_mut() {
                    view_dest
                        .entities
                        .append(&mut thread_entity_queue[view_dest_index]);
                }
                view_dest.shrink();
                view_dest.entities.sort_unstable();
            }

            // MPS: each cached stable cascade's view: its live casters' count
            // and tiles, its settled casters' list (or print) and revision.
            for (slot, far) in walks.iter().enumerate() {
                let view_cache = light_cache
                    .as_mut()
                    .unwrap()
                    .bypass_change_detection()
                    .views
                    .get_mut(&(*view, far.cascade as u32))
                    .and_then(|state| state.view.as_mut())
                    .unwrap();
                let statics = &mut *far_scratch;
                statics.clear();
                let mut tiles = mem::take(&mut view_cache.movers_tiles);
                tiles.reset(map_size);
                let mut print = StaticCastersPrint::default();
                let mut live = 0;
                let mut reference_entities = Vec::new();
                for local in far_queue
                    .iter_mut()
                    .filter_map(|locals| locals.get_mut(slot))
                    .filter(|local| local.used)
                {
                    local.used = false;
                    statics.append(&mut local.statics);
                    reference_entities.append(&mut local.reference);
                    tiles.union_with(&local.tiles);
                    print.merge(mem::take(&mut local.print));
                    live += local.live;
                }
                view_cache.live_casters = live;
                view_cache.reference = reference.then(|| {
                    reference_entities.sort_unstable();
                    reference_entities
                });
                if far.single_view {
                    // Its settled casters count only for the revision, by an
                    // order-free print; its tiles are all set while a live
                    // caster is in it. The split views' list is found again (a
                    // new revision) when it has a usable layer again.
                    tiles.reset(map_size);
                    if live > 0 {
                        tiles.fill();
                    }
                    if view_cache.static_print != Some(print) {
                        if !far.invalidated {
                            view_cache.static_revision += 1;
                        }
                        view_cache.static_print = Some(print);
                    }
                    view_cache.static_entities.clear();
                    view_cache.statics_walked = false;
                } else {
                    view_cache.static_print = None;
                    view_cache.statics_walked = far.walk_statics;
                    if far.walk_statics {
                        statics.sort_unstable();
                        if *statics != view_cache.static_entities {
                            // A change that wasn't reported (or the first list).
                            if !far.invalidated {
                                view_cache.static_revision += 1;
                            }
                            mem::swap(&mut view_cache.static_entities, statics);
                        }
                    }
                }
                view_cache.movers_tiles = tiles;
            }
        }
    }

    if !(settling.is_empty() && stock.is_empty())
        && let Ok(mut counts) = shadow_cache_feedback.1.lock()
    {
        for id in settling {
            counts.entry(id).or_default().settling += 1;
        }
        for id in stock {
            counts.entry(id).or_default().stock += 1;
        }
    }

    // Defer marking view visibility so this system can run in parallel with check_point_light_mesh_visibility
    // TODO: use resource to avoid unnecessary memory alloc
    let mut defer_queue = mem::take(defer_visible_entities_queue.deref_mut());
    commands.queue(move |world: &mut World| {
        let mut query = world.query::<&mut ViewVisibility>();
        for entities in defer_queue.iter_mut() {
            let mut iter = query.iter_many_mut(world, entities.iter()).matched();
            while let Some(mut view_visibility) = iter.fetch_next() {
                view_visibility.set_visible();
            }
        }
    });
}

/// MPS: the stand-ins cascade `index` of `cascade_count` draws instead of the
/// detail (the fault: the middle cascade takes the far one's).
fn cascade_selects(
    policy: &StableCascades,
    index: usize,
    cascade_count: usize,
    middle_stand_in_routing: bool,
) -> Option<u32> {
    match policy.selects_stand_ins(index, cascade_count) {
        Some(_) if middle_stand_in_routing && index + 2 == cascade_count => Some(index as u32 + 1),
        selects => selects,
    }
}

/// MPS: how [`check_dir_light_mesh_visibility`] walks a camera's cached
/// stable cascade this frame.
struct FarWalk {
    cascade: usize,
    clip_from_world: bevy_math::Mat4,
    /// No usable static layer: every caster into one list.
    single_view: bool,
    /// Its settled casters are visited (their list is drawn this frame).
    walk_statics: bool,
    invalidated: bool,
}

/// MPS: a thread's share of a cached stable cascade's walk in
/// [`check_dir_light_mesh_visibility`].
#[derive(Default)]
#[doc(hidden)]
pub struct FarCascadeQueue {
    used: bool,
    statics: Vec<Entity>,
    reference: Vec<Entity>,
    tiles: ShadowTileMask,
    print: StaticCastersPrint,
    live: u32,
}

/// Updates the visibility for [`PointLight`]s and [`SpotLight`]s so that shadow
/// map rendering can work.
///
/// This only processes entities without [`NoCpuCulling`]. Entities with
/// [`NoCpuCulling`] receive no view-specific processing in the main world.
///
/// A cached spot light (see [`SpotLightShadowCache::is_cached`]) lists only its
/// live casters in its [`VisibleMeshEntities`], every frame, with the tiles of
/// its shadow map their [`ShadowFootprint`]s touch ([`ShadowTileMask`]). Its
/// settled casters ([`StaticShadowCaster`]) go to its [`SpotLightShadowCache`],
/// and are visited only when the render world may draw them: when they or the
/// light changed, until the render world acknowledged a shadow map drawn from
/// them, and every frame while the light has no static layer.
pub fn check_point_light_mesh_visibility(
    visible_point_lights: Query<(
        &VisibleEntities,
        Option<&Frustum>,
        Option<&Camera>,
        Has<Clusters>,
    )>,
    mut point_lights: Query<(
        &PointLight,
        &GlobalTransform,
        &CubemapFrusta,
        &mut CubemapVisibleEntities,
        Option<&RenderLayers>,
    )>,
    mut spot_lights: Query<(
        &SpotLight,
        &GlobalTransform,
        &Frustum,
        &mut VisibleMeshEntities,
        Option<&RenderLayers>,
        Option<Mut<SpotLightShadowCache>>,
        Has<ShadowCacheExempt>,
    )>,
    mut movable_casters: PointLightShadowCasterQuery<Without<StaticShadowCaster>>,
    mut static_casters: PointLightShadowCasterQuery<With<StaticShadowCaster>>,
    mut camera_query: Query<(Entity, &RenderTarget), With<Camera>>,
    mut shadow_lod_origin_query: Query<Entity, With<ShadowLodOrigin>>,
    mut point_and_spot_light_query: Query<Entity, Or<(With<PointLight>, With<SpotLight>)>>,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    (directional_light_shadow_map, shadow_cache_feedback, shadow_cache_settings, time, faults): (
        Res<DirectionalLightShadowMap>,
        Res<ShadowCacheFeedback>,
        Res<ShadowCacheSettings>,
        Option<Res<bevy_time::Time>>,
        Option<Res<ShadowCacheFaults>>,
    ),
    mut cubemap_visible_entities_queue: Local<Parallel<[Vec<Entity>; 6]>>,
    mut spot_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
    (mut movers_grid, mut mover_candidates, mut movers_queue, mover_bounds): (
        Local<MoversGrid>,
        Local<Vec<Entity>>,
        Local<Parallel<(Vec<Entity>, ShadowTileMask, bool)>>,
        MoverBoundsQuery,
    ),
    (mut checked_lights, mut frame, mut grid_found_sparse): (
        Local<EntityHashSet>,
        Local<u32>,
        Local<bool>,
    ),
    (mut light_views, mut list_scratch, mut static_prints): (
        Local<LightViews>,
        Local<Vec<Entity>>,
        Local<Parallel<StaticCastersPrint>>,
    ),
) {
    checked_lights.clear();
    *frame = frame.wrapping_add(1);
    let frame = *frame;
    let now = time.map_or(core::time::Duration::ZERO, |time| time.elapsed());
    let stale_ack = faults.is_some_and(|faults| faults.stale_ack);
    // The movers grid, built at the first cached spot light of a frame that
    // uses it: while a light had few movers near it when it was last built,
    // and every 32 frames to see whether one has.
    let use_grid = *grid_found_sparse || frame.is_multiple_of(32);
    let mut movers_grid_built = false;
    let mut found_sparse = false;
    // A cached light walks a few hundred movers serially: below that, the
    // tasks of a parallel walk cost more than the walk.
    let few_movers = movable_casters.iter().len() < 256;

    let shadow_lod_origin = get_shadow_lod_origin(
        camera_query.transmute_lens_filtered(),
        shadow_lod_origin_query.transmute_lens_filtered(),
        point_and_spot_light_query.transmute_lens_filtered(),
    );

    // MPS: the render world's reports on cached shadow views, as of the
    // frames it finished.
    let shadow_cache_status = shadow_cache_feedback.views();

    let visible_entity_ranges = visible_entity_ranges.as_deref();
    // MPS: the active clustered cameras each light is visible to (by their
    // frusta), once per frame: whether a cached light lights a view.
    light_views.build(&visible_point_lights);
    for (visible_lights, ..) in &visible_point_lights {
        for &light_entity in visible_lights.get(TypeId::of::<ClusterVisibilityClass>()) {
            if !checked_lights.insert(light_entity) {
                continue;
            }

            // Point lights
            if let Ok((
                point_light,
                transform,
                cubemap_frusta,
                mut cubemap_visible_entities,
                maybe_view_mask,
            )) = point_lights.get_mut(light_entity)
            {
                // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
                if !point_light.shadow_maps_enabled {
                    continue;
                }

                let view_mask = maybe_view_mask.unwrap_or_default();
                let light_sphere = Sphere {
                    center: Vec3A::from(transform.translation()),
                    radius: point_light.range,
                };

                let check_caster =
                    |cubemap_visible_entities_local_queue: &mut [Vec<Entity>; 6],
                     item: PointLightShadowCasterItem| {
                        let (
                            entity,
                            inherited_visibility,
                            mut view_visibility,
                            maybe_entity_mask,
                            maybe_aabb,
                            maybe_transform,
                            has_visibility_range,
                            has_no_frustum_culling,
                            (.., representation),
                        ) = item;
                        // MPS: a far cascade's stand-ins cast only into far
                        // cascades.
                        if !inherited_visibility.get() || !view_takes(None, representation.copied())
                        {
                            return;
                        }
                        let entity_mask = maybe_entity_mask.unwrap_or_default();
                        if !view_mask.intersects(entity_mask) {
                            return;
                        }
                        if has_visibility_range
                            && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                                shadow_lod_origin.is_none_or(|shadow_lod_origin| {
                                    !visible_entity_ranges
                                        .entity_is_in_range_of_view(entity, shadow_lod_origin)
                                })
                            })
                        {
                            return;
                        }

                        // If we have an aabb and transform, do frustum culling
                        if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                            let model_to_world = transform.affine();
                            // Do a cheap sphere vs obb test to prune out most meshes outside the sphere of the light
                            if !has_no_frustum_culling
                                && !light_sphere.intersects_obb(&aabb, &model_to_world)
                            {
                                return;
                            }

                            for (frustum, visible_entities) in cubemap_frusta
                                .iter()
                                .zip(cubemap_visible_entities_local_queue.iter_mut())
                            {
                                if has_no_frustum_culling
                                    || frustum.intersects_obb(&aabb, &model_to_world, true, true)
                                {
                                    view_visibility.set_visible();
                                    visible_entities.push(entity);
                                }
                            }
                        } else {
                            view_visibility.set_visible();
                            for visible_entities in cubemap_visible_entities_local_queue.iter_mut()
                            {
                                visible_entities.push(entity);
                            }
                        }
                    };
                // MPS: static and movable casters are separate queries; point
                // lights draw both.
                movable_casters.par_iter_mut().for_each_init(
                    || cubemap_visible_entities_queue.borrow_local_mut(),
                    |queue, item| check_caster(queue, item),
                );
                static_casters.par_iter_mut().for_each_init(
                    || cubemap_visible_entities_queue.borrow_local_mut(),
                    |queue, item| check_caster(queue, item),
                );

                // Collect the entities from each parallel queue.
                for (view_dest_index, view_dest) in cubemap_visible_entities.iter_mut().enumerate()
                {
                    view_dest.entities.clear();
                    for thread_entity_queue in cubemap_visible_entities_queue.iter_mut() {
                        view_dest
                            .entities
                            .append(&mut thread_entity_queue[view_dest_index]);
                    }
                    view_dest.shrink();
                    view_dest.entities.sort_unstable();
                }
            }

            // Spot lights
            if let Ok((
                spot_light,
                transform,
                frustum,
                mut visible_entities,
                maybe_view_mask,
                mut shadow_cache,
                exempt,
            )) = spot_lights.get_mut(light_entity)
            {
                let view_mask = maybe_view_mask.unwrap_or_default();
                // MPS: whether the light is cached: the cache on, shadows on,
                // and what projects its shadow map settled. A light that
                // changed is drawn whole every frame until it settles.
                let projection = SpotShadowProjection::new(
                    transform,
                    spot_light,
                    view_mask,
                    directional_light_shadow_map.size,
                );
                let cached = shadow_cache.as_mut().is_some_and(|cache| {
                    let settled = cache.bypass_change_detection().settlement.settle(
                        &projection,
                        frame,
                        now,
                        shadow_cache_settings.settle_time,
                    );
                    shadow_cache_settings.enabled
                        && spot_light.shadow_maps_enabled
                        && !exempt
                        && settled
                });
                if let Some(cache) = shadow_cache.as_mut()
                    && cache.cached != cached
                {
                    cache.cached = cached;
                    cache.view.live_casters = 0;
                    if !cached {
                        // Drawn again from scratch once it is cached again.
                        cache.view.projection = None;
                    }
                }

                // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
                if !spot_light.shadow_maps_enabled {
                    continue;
                }

                let light_sphere = Sphere {
                    center: Vec3A::from(transform.translation()),
                    radius: spot_light.range,
                };

                // Whether a caster is in the light's frustum (or has no
                // bounds); marks it visible if so.
                let caster_is_visible = |item: &mut PointLightShadowCasterItem| -> bool {
                    let (
                        entity,
                        inherited_visibility,
                        view_visibility,
                        maybe_entity_mask,
                        maybe_aabb,
                        maybe_transform,
                        has_visibility_range,
                        has_no_frustum_culling,
                        (.., live_material, representation),
                    ) = item;
                    // MPS: a far cascade's stand-ins cast only into far
                    // cascades.
                    if !inherited_visibility.get() || !view_takes(None, representation.copied()) {
                        return false;
                    }

                    let entity_mask = maybe_entity_mask.unwrap_or_default();
                    if !view_mask.intersects(entity_mask) {
                        return false;
                    }
                    // Check visibility ranges.
                    if *has_visibility_range
                        && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                            shadow_lod_origin.is_none_or(|shadow_lod_origin| {
                                !visible_entity_ranges
                                    .entity_is_in_range_of_view(*entity, shadow_lod_origin)
                            })
                        })
                    {
                        return false;
                    }

                    // MPS: the bounds grown by what a live material can
                    // displace; a caster whose grown bounds aren't known is
                    // visible as one without bounds.
                    let padded = match (maybe_aabb, maybe_transform) {
                        (Some(aabb), Some(transform)) => {
                            padded_aabb(aabb, &transform.affine(), *live_material)
                                .map(|aabb| (aabb, transform.affine()))
                        }
                        _ => None,
                    };
                    if let Some((aabb, model_to_world)) = &padded {
                        // Do a cheap sphere vs obb test to prune out most meshes outside the sphere of the light
                        if !*has_no_frustum_culling
                            && !light_sphere.intersects_obb(aabb, model_to_world)
                        {
                            return false;
                        }

                        if *has_no_frustum_culling
                            || frustum.intersects_obb(aabb, model_to_world, true, true)
                        {
                            view_visibility.set_visible();
                            return true;
                        }
                        false
                    } else {
                        view_visibility.set_visible();
                        true
                    }
                };

                let Some(mut light_cache) = shadow_cache.filter(|_| cached) else {
                    let check_caster =
                        |spot_visible_entities_local_queue: &mut Vec<Entity>,
                         mut item: PointLightShadowCasterItem| {
                            if caster_is_visible(&mut item) {
                                spot_visible_entities_local_queue.push(item.0);
                            }
                        };
                    movable_casters.par_iter_mut().for_each_init(
                        || spot_visible_entities_queue.borrow_local_mut(),
                        |queue, item| check_caster(queue, item),
                    );
                    static_casters.par_iter_mut().for_each_init(
                        || spot_visible_entities_queue.borrow_local_mut(),
                        |queue, item| check_caster(queue, item),
                    );

                    visible_entities.entities.clear();
                    for thread_entity_queue in spot_visible_entities_queue.iter_mut() {
                        visible_entities.entities.append(thread_entity_queue);
                    }
                    visible_entities.shrink();
                    visible_entities.entities.sort_unstable();
                    continue;
                };

                // MPS: a cached spot light: its one shadow view. Its first
                // frame cached draws its static layer.
                let view_id = ShadowViewId::spot(light_entity);
                let cache = &mut light_cache.view;
                cache.begin();
                if !cache
                    .projection
                    .as_ref()
                    .is_some_and(|cached| cached.is_spot(&projection))
                {
                    cache.invalidated = true;
                }
                let invalidated = mem::take(&mut cache.invalidated);
                if invalidated {
                    cache.static_revision += 1;
                    cache.projection = Some(ShadowViewProjection::Spot(projection.clone()));
                }
                let clip_from_world = projection.clip_from_world();
                let near_z = projection.near_z();

                // MPS: a light that reaches into no view (its range misses the
                // frustum of every active clustered camera that sees it, as
                // clustering tests it) shades nothing: it finds no casters,
                // and the render world draws nothing for it.
                cache.lights_a_view = light_views.lights_a_view(light_entity, &light_sphere);

                let map_size = directional_light_shadow_map.size as u32;
                // A light without a static layer is drawn whole whenever a
                // live caster is in its frustum: where they fall doesn't
                // matter.
                let status = cache.status(&shadow_cache_status, &view_id);
                cache.single_view = status.is_some_and(|status| status.fallback);
                if !cache.lights_a_view {
                    cache.live_casters = 0;
                    cache.movers_tiles.reset(map_size);
                    cache.statics_walked = false;
                    if !visible_entities.entities.is_empty() {
                        visible_entities.entities.clear();
                    }
                    continue;
                }
                // A light without a usable static layer: walked as an uncached
                // light, all its casters into one list for its one view. No
                // footprints: its tiles are all set while a live caster is in
                // its frustum. Its settled casters count only for the
                // revision, by an order-free print instead of a list: a change
                // no one reported redraws it.
                if cache.single_view {
                    list_scratch.clear();
                    movable_casters.par_iter_mut().for_each_init(
                        || spot_visible_entities_queue.borrow_local_mut(),
                        |queue, mut item| {
                            if caster_is_visible(&mut item) {
                                queue.push(item.0);
                            }
                        },
                    );
                    for queue in spot_visible_entities_queue.iter_mut() {
                        list_scratch.append(queue);
                    }
                    let live = list_scratch.len();
                    static_casters.par_iter_mut().for_each_init(
                        || {
                            (
                                spot_visible_entities_queue.borrow_local_mut(),
                                static_prints.borrow_local_mut(),
                            )
                        },
                        |(queue, print), mut item| {
                            if caster_is_visible(&mut item) {
                                queue.push(item.0);
                                print.add(item.0);
                            }
                        },
                    );
                    for queue in spot_visible_entities_queue.iter_mut() {
                        list_scratch.append(queue);
                    }
                    let mut print = StaticCastersPrint::default();
                    for thread_print in static_prints.iter_mut() {
                        print.merge(mem::take(thread_print));
                    }
                    list_scratch.sort_unstable();
                    if visible_entities.entities != *list_scratch {
                        mem::swap(&mut visible_entities.entities, &mut *list_scratch);
                    }
                    if cache.static_print != Some(print) {
                        if !invalidated {
                            cache.static_revision += 1;
                        }
                        cache.static_print = Some(print);
                    }
                    // The split views' list is found again (a new revision)
                    // when the light has a usable layer again.
                    cache.static_entities.clear();
                    cache.statics_walked = false;
                    cache.live_casters = live as u32;
                    cache.movers_tiles.reset(map_size);
                    if live > 0 {
                        cache.movers_tiles.fill();
                    }
                    continue;
                }
                cache.static_print = None;
                // Movable casters, every frame, with where they fall in the
                // shadow map, by the same test as every other caster: walked
                // all in parallel, or, while most lights have few near them,
                // only those the grid puts near the light.
                if use_grid && !movers_grid_built {
                    movers_grid_built = true;
                    movers_grid.build(
                        mover_bounds
                            .iter()
                            .filter(|(_, inherited_visibility, ..)| inherited_visibility.get())
                            .map(|(entity, _, aabb, transform, no_frustum_culling, live)| {
                                let bounds = match (aabb, transform) {
                                    (Some(aabb), Some(transform)) if !no_frustum_culling => {
                                        let (center, half_extents) =
                                            MoversGrid::world_bounds(aabb, transform);
                                        let padding =
                                            live.map_or(0.0, |live| live.max_displacement);
                                        Some((center, half_extents + padding))
                                    }
                                    _ => None,
                                };
                                (entity, bounds)
                            }),
                    );
                }
                mover_candidates.clear();
                if use_grid {
                    movers_grid.candidates(
                        light_sphere.center,
                        light_sphere.radius,
                        &mut *mover_candidates,
                    );
                }
                // (A candidate looked up costs about four movers walked.)
                let walk_all = !use_grid || mover_candidates.len() * 4 >= movers_grid.len();
                found_sparse |= !walk_all;
                // Where a visible mover falls in the shadow map.
                let footprint = |item: &PointLightShadowCasterItem| -> ShadowFootprint {
                    let (skinned, dynamic_skinned_bounds, mesh_morphed, morphed, live, _) = item.8;
                    match (&item.4, item.5) {
                        // Bounds that may not hold the mesh this frame:
                        // skinned without updated bounds, morphed, or drawn
                        // wherever it is.
                        (Some(aabb), Some(transform))
                            if !item.7
                                && !(mesh_morphed || morphed)
                                && (!skinned || (dynamic_skinned_bounds && aabb.is_changed())) =>
                        {
                            let world_from_local = transform.affine();
                            padded_aabb(aabb, &world_from_local, live).map_or(
                                ShadowFootprint::Whole,
                                |aabb| {
                                    ShadowFootprint::of_box(
                                        &clip_from_world,
                                        near_z,
                                        &aabb,
                                        &world_from_local,
                                    )
                                },
                            )
                        }
                        _ => ShadowFootprint::Whole,
                    }
                };
                // Each thread with a list and tiles of its own (cleared at its
                // first mover of the light), merged after.
                let visit = |queue: &mut (Vec<Entity>, ShadowTileMask, bool),
                             mut item: PointLightShadowCasterItem| {
                    if !caster_is_visible(&mut item) {
                        return;
                    }
                    let (entities, tiles, used) = queue;
                    if !*used {
                        *used = true;
                        tiles.reset(map_size);
                    }
                    entities.push(item.0);
                    tiles.insert(footprint(&item));
                };
                if walk_all && !few_movers {
                    // Most movers are near: a walk over all of them in parallel
                    // (the same test rejects the others) beats looking each
                    // candidate up.
                    movable_casters.par_iter_mut().for_each_init(
                        || movers_queue.borrow_local_mut(),
                        |queue, item| visit(queue, item),
                    );
                } else if walk_all {
                    let mut queue = movers_queue.borrow_local_mut();
                    for item in movable_casters.iter_mut() {
                        visit(&mut queue, item);
                    }
                } else {
                    let mut queue = movers_queue.borrow_local_mut();
                    for &entity in mover_candidates.iter() {
                        if let Ok(item) = movable_casters.get_mut(entity) {
                            visit(&mut queue, item);
                        }
                    }
                }
                let movers = &mut *list_scratch;
                movers.clear();
                let mut movers_tiles = mem::take(&mut cache.movers_tiles);
                movers_tiles.reset(map_size);
                for (entities, tiles, used) in movers_queue.iter_mut().filter(|queue| queue.2) {
                    movers.append(entities);
                    movers_tiles.union_with(tiles);
                    *used = false;
                }
                movers.sort_unstable();
                cache.live_casters = movers.len() as u32;
                // Touch the list only when it changes.
                if visible_entities.entities != *movers {
                    mem::swap(&mut visible_entities.entities, movers);
                }

                // Static casters, only when the render world may draw them:
                // a new revision until it acknowledged a map drawn from them.
                let walk_statics = invalidated
                    || status.is_none_or(|status| {
                        let acknowledged = status.acknowledged_static_revision;
                        acknowledged != Some(cache.static_revision)
                            // The fault: the previous revision's
                            // acknowledgement taken for this one.
                            && !(stale_ack && acknowledged == Some(cache.static_revision - 1))
                    });
                cache.movers_tiles = movers_tiles;
                cache.statics_walked = walk_statics;
                if walk_statics {
                    static_casters.par_iter_mut().for_each_init(
                        || spot_visible_entities_queue.borrow_local_mut(),
                        |entities, mut item| {
                            if caster_is_visible(&mut item) {
                                entities.push(item.0);
                            }
                        },
                    );
                    let statics = &mut *list_scratch;
                    statics.clear();
                    for thread_entity_queue in spot_visible_entities_queue.iter_mut() {
                        statics.append(thread_entity_queue);
                    }
                    statics.sort_unstable();
                    if *statics != cache.static_entities {
                        // A change that wasn't reported (or the first list).
                        if !invalidated {
                            cache.static_revision += 1;
                        }
                        mem::swap(&mut cache.static_entities, statics);
                    }
                }
            }
        }
    }
    if use_grid {
        *grid_found_sparse = found_sparse;
    }
}

/// MPS: for [`check_point_light_mesh_visibility`], the active clustered
/// cameras each light is visible to, built once a frame.
#[derive(Default)]
#[doc(hidden)]
pub struct LightViews {
    /// The frusta of the active clustered cameras (`None`: no frustum).
    frusta: Vec<Option<Frustum>>,
    /// By light, the indices of the cameras it is visible to.
    by_light: EntityHashMap<SmallVec<[u32; 2]>>,
}

impl LightViews {
    fn build(
        &mut self,
        views: &Query<(
            &VisibleEntities,
            Option<&Frustum>,
            Option<&Camera>,
            Has<Clusters>,
        )>,
    ) {
        self.frusta.clear();
        self.by_light.clear();
        for (visible, frustum, camera, clustered) in views {
            if !(clustered && camera.is_some_and(|camera| camera.is_active)) {
                continue;
            }
            let index = self.frusta.len() as u32;
            self.frusta.push(frustum.copied());
            for &light in visible.get(TypeId::of::<ClusterVisibilityClass>()) {
                self.by_light.entry(light).or_default().push(index);
            }
        }
    }

    /// Whether a light reaches into a view: it is visible to an active
    /// clustered camera whose frustum its range meets (as clustering tests
    /// it).
    fn lights_a_view(&self, light: Entity, range: &Sphere) -> bool {
        self.by_light.get(&light).is_some_and(|views| {
            views.iter().any(|&index| {
                self.frusta[index as usize]
                    .is_none_or(|frustum| frustum.intersects_sphere(range, true))
            })
        })
    }
}

/// MPS: an order-free print of a set of casters (a count, a sum and a xor of
/// mixed entity bits), merged across threads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct StaticCastersPrint {
    count: u64,
    sum: u64,
    xor: u64,
}

impl StaticCastersPrint {
    fn add(&mut self, entity: Entity) {
        // SplitMix64's finalizer, twice with different seeds.
        let mix = |mut z: u64| {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let bits = entity.to_bits();
        self.count += 1;
        self.sum = self.sum.wrapping_add(mix(bits));
        self.xor ^= mix(bits ^ 0x9e37_79b9_7f4a_7c15);
    }

    fn merge(&mut self, other: StaticCastersPrint) {
        self.count += other.count;
        self.sum = self.sum.wrapping_add(other.sum);
        self.xor ^= other.xor;
    }
}

/// The meshes that [`check_point_light_mesh_visibility`] tests against the
/// frusta of point and spot lights: either the static or the movable ones
/// (`F`).
type PointLightShadowCasterQuery<'w, 's, F> = Query<
    'w,
    's,
    (
        Entity,
        &'static InheritedVisibility,
        &'static mut ViewVisibility,
        Option<&'static RenderLayers>,
        Option<Ref<'static, Aabb>>,
        Option<&'static GlobalTransform>,
        Has<VisibilityRange>,
        Has<NoFrustumCulling>,
        (
            Has<SkinnedMesh>,
            Has<DynamicSkinnedMeshBounds>,
            Has<MeshMorphWeights>,
            Has<MorphWeights>,
            Option<&'static LiveShadowMaterial>,
            Option<&'static ShadowRepresentation>,
        ),
    ),
    (
        Without<NotShadowCaster>,
        Without<DirectionalLight>,
        Without<NoCpuCulling>,
        With<Mesh3d>,
        F,
    ),
>;

/// The bounds of the movable casters that [`check_point_light_mesh_visibility`]
/// puts into its [`MoversGrid`]: those of [`PointLightShadowCasterQuery`]
/// without [`StaticShadowCaster`].
type MoverBoundsQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static InheritedVisibility,
        Option<&'static Aabb>,
        Option<&'static GlobalTransform>,
        Has<NoFrustumCulling>,
        Option<&'static LiveShadowMaterial>,
    ),
    (
        Without<NotShadowCaster>,
        Without<DirectionalLight>,
        Without<NoCpuCulling>,
        With<Mesh3d>,
        Without<StaticShadowCaster>,
    ),
>;

/// An item of [`PointLightShadowCasterQuery`].
type PointLightShadowCasterItem<'w> = (
    Entity,
    &'w InheritedVisibility,
    Mut<'w, ViewVisibility>,
    Option<&'w RenderLayers>,
    Option<Ref<'w, Aabb>>,
    Option<&'w GlobalTransform>,
    bool,
    bool,
    (
        bool,
        bool,
        bool,
        bool,
        Option<&'w LiveShadowMaterial>,
        Option<&'w ShadowRepresentation>,
    ),
);

/// MPS: a caster's bounds grown by what its live material can displace a
/// vertex, in world units (see [`LiveShadowMaterial`]); `None` when no local
/// box holds the displaced mesh (a singular or nonfinite transform): the
/// caster is then unbounded.
///
/// A world displacement `d` moves a vertex by `local_from_world * d` in local
/// space, so local axis `i` grows by the padding times the length of row `i`
/// of the inverse: exact for any transform, shear included.
fn padded_aabb(
    aabb: &Aabb,
    world_from_local: &bevy_math::Affine3A,
    live: Option<&LiveShadowMaterial>,
) -> Option<Aabb> {
    let Some(padding) = live
        .map(|live| live.max_displacement)
        .filter(|padding| *padding > 0.0)
    else {
        return Some(*aabb);
    };
    let axes = world_from_local.matrix3;
    let determinant = axes.determinant();
    if !(determinant.is_finite() && determinant != 0.0) {
        return None;
    }
    let local_from_world = axes.inverse().transpose();
    let grow = Vec3A::new(
        local_from_world.x_axis.length(),
        local_from_world.y_axis.length(),
        local_from_world.z_axis.length(),
    ) * padding;
    let half_extents = aabb.half_extents + grow;
    half_extents.is_finite().then_some(Aabb {
        center: aabb.center,
        half_extents,
    })
}

/// Determines the LOD origin for spot and point light shadow maps.
///
/// The selection priority is, from highest to lowest:
///
/// 1. An entity explicitly marked with the [`ShadowLodOrigin`] component.
///
/// 2. A camera that renders to a window.
///
/// 3. Any camera.
pub fn get_shadow_lod_origin(
    mut camera_query: QueryLens<(Entity, &RenderTarget), With<Camera>>,
    mut shadow_lod_origin_query: QueryLens<Entity, With<ShadowLodOrigin>>,
    mut lights_query: QueryLens<Entity, Or<(With<PointLight>, With<SpotLight>)>>,
) -> Option<Entity> {
    let (camera_query, shadow_lod_origin_query) =
        (camera_query.query(), shadow_lod_origin_query.query());

    let mut entities: SmallVec<[Entity; 4]> = smallvec![];
    entities.extend(shadow_lod_origin_query.iter());
    if let Some(lod_origin) = entities.iter().min() {
        return Some(*lod_origin);
    }

    entities.extend(
        camera_query
            .iter()
            .filter_map(|(main_entity, render_target)| match *render_target {
                RenderTarget::Window(_) => Some(main_entity),
                _ => None,
            }),
    );
    if let Some(lod_origin) = entities.iter().min() {
        return Some(*lod_origin);
    };

    entities.extend(camera_query.iter().map(|(main_entity, _)| main_entity));
    if let Some(lod_origin) = entities.iter().min() {
        if !lights_query.query().is_empty() {
            warn_once!(
                "Point lights and/or spot lights are present, but no entity has \
                 `ShadowLodOrigin`, and no camera that renders to the window has been found. \
                 Consider using the `ShadowLodOrigin` component to set a LOD origin."
            );
        }

        return Some(*lod_origin);
    };

    None
}
