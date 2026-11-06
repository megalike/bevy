//! MPS: a stable projection for a directional light's far cascade, opt-in
//! ([`StableFarCascade`]).
//!
//! Bevy fits every cascade to the camera's frustum slice each frame: its
//! center follows the camera's rotation and its depth mapping the slice's
//! depth, so a cached map of the cascade could never be reused. The stable
//! projection covers the whole sphere around the camera that the far slice
//! can reach in any orientation (the far plane's corners), plus a guard, so a
//! camera that turns, cuts or looks up keeps the same projection; only a
//! translation of the camera beyond the guard re-centers it. Its center is
//! snapped to whole texels, and its depth maps a fixed light-space interval
//! that covers the scene's conservative bounds ([`ShadowSceneBounds`]),
//! rounded outward to power-of-two bins: the mapping changes only when the
//! scene grows past the interval or the sun turns.
//!
//! The policy runs the same whether the shadow cache is on or off: it decides
//! the projection, never the cache.

use bevy_camera::{
    primitives::Aabb,
    visibility::{NoCpuCulling, NotCameraVisible},
    Projection,
};
use bevy_ecs::{entity::EntityHashMap, lifecycle::HookContext, prelude::*, world::DeferredWorld};
use bevy_log::warn_once;
use bevy_math::{ops, Mat4, Quat, Vec2, Vec3A, Vec4};
use bevy_mesh::Mesh3d;
use bevy_reflect::prelude::*;
use bevy_transform::components::GlobalTransform;

use crate::cascade::Cascade;

/// Opt-in: a [`DirectionalLight`](crate::DirectionalLight)'s last cascade
/// takes a stable projection that only a camera translation beyond
/// [`Self::guard`] moves (see the module docs). The other cascades keep
/// Bevy's projection. Needs at least two cascades; with one, the policy does
/// nothing (its one cascade covers close contacts too).
///
/// The stable cascade covers a larger area than Bevy's fitted one, so its
/// texels are larger (about `2 (1 + guard) r / D₀` times, `r` being the
/// distance to the slice's far corners and `D₀` Bevy's diameter).
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
#[require(StableFarProjections)]
pub struct StableFarCascade {
    /// How far the camera may move from the projection's center before it
    /// re-centers, as a share of the coverage radius (the distance from the
    /// camera to the far slice's corners).
    pub guard: f32,
    /// Texels kept at every edge beyond the coverage, for the receivers'
    /// filter taps and normal bias.
    pub filter_texels: f32,
    /// The far cascade draws the casters marked as its stand-ins
    /// ([`ShadowRepresentation::FarCascade`](crate::ShadowRepresentation))
    /// instead of those marked as their detail
    /// ([`ShadowRepresentation::Detail`](crate::ShadowRepresentation)).
    /// Unmarked casters cast into every cascade either way.
    pub stand_ins: bool,
}

/// The retained far projection of each camera of a light with
/// [`StableFarCascade`] (added with it).
#[derive(Component, Clone, Debug, Default)]
pub struct StableFarProjections {
    pub(crate) retained: EntityHashMap<RetainedFarProjection>,
}

impl Default for StableFarCascade {
    fn default() -> Self {
        Self {
            guard: 0.125,
            filter_texels: 8.0,
            stand_ins: false,
        }
    }
}

impl StableFarProjections {
    /// How many times a camera's far projection was (re)built since the
    /// light got the policy: a new sun direction, a re-center, a new depth
    /// interval or a new camera projection. For tools and tests.
    pub fn rebuilds(&self, camera: Entity) -> Option<u64> {
        self.retained.get(&camera).map(|retained| retained.rebuilds)
    }
}

/// MPS: which shadow views a mesh casts into, besides its render layers: a
/// mesh can stand in for others in the far cascade of lights that select
/// stand-ins ([`StableFarCascade::stand_ins`]). Meshes without it cast into
/// every shadow view.
///
/// | View | [`Self::Detail`] | [`Self::FarCascadeStandIn`] |
/// | --- | --- | --- |
/// | cameras | drawn | not drawn ([`NotCameraVisible`], inserted with it) |
/// | far cascade of a light selecting stand-ins | not drawn | drawn |
/// | other cascades, spot and point lights | drawn | not drawn |
///
/// The selection belongs to the shadow view, never to the shadow cache: a
/// view draws the same casters with the cache on, off or falling back.
/// Changing it makes a settled caster live again, like a change of render
/// layers. Only the CPU culling path honors it (not with [`NoCpuCulling`]).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
#[reflect(Component, Default, Debug, Clone, PartialEq)]
#[component(on_insert = on_insert_shadow_representation)]
pub enum ShadowRepresentation {
    /// The detail a far cascade's stand-ins replace (terrain patches the
    /// camera selects): cast into every shadow view except the far cascades
    /// of lights that select stand-ins.
    #[default]
    Detail,
    /// A stand-in cast only into the far cascades of lights that select
    /// stand-ins; never drawn by cameras.
    FarCascadeStandIn,
}

fn on_insert_shadow_representation(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    if world.get::<NoCpuCulling>(entity).is_some() {
        warn_once!(
            "ShadowRepresentation on an entity with NoCpuCulling is ignored on the GPU culling path"
        );
    }
    if world.get::<ShadowRepresentation>(entity) == Some(&ShadowRepresentation::FarCascadeStandIn)
        && world.get::<NotCameraVisible>(entity).is_none()
    {
        world.commands().entity(entity).insert(NotCameraVisible);
    }
}

/// Whether a shadow view draws a caster of this representation (`None`:
/// unmarked). `selects_stand_ins`: the view is the far cascade of a light
/// that selects stand-ins.
pub(crate) fn view_takes(
    selects_stand_ins: bool,
    representation: Option<ShadowRepresentation>,
) -> bool {
    match representation {
        None => true,
        Some(ShadowRepresentation::Detail) => !selects_stand_ins,
        Some(ShadowRepresentation::FarCascadeStandIn) => selects_stand_ins,
    }
}

/// The world-space bounds of every mesh, grown as meshes move or appear and
/// never shrunk (until [`Self::reset`]): the depth interval of stable far
/// cascades covers them. Kept only while a light has [`StableFarCascade`].
#[derive(Resource, Clone, Debug, Default)]
pub struct ShadowSceneBounds {
    bounds: Option<(Vec3A, Vec3A)>,
    /// The next update walks every mesh, not only the changed ones.
    full: bool,
}

impl ShadowSceneBounds {
    /// The bounds' minimum and maximum corners, if any mesh has bounds.
    pub fn bounds(&self) -> Option<(Vec3A, Vec3A)> {
        self.bounds
    }

    /// Forgets the bounds; they are found again from every mesh in the next
    /// update (a new scene). Stable far cascades keep their depth interval
    /// while it still covers the new bounds.
    pub fn reset(&mut self) {
        self.bounds = None;
        self.full = true;
    }

    /// Grows the bounds by a box (minimum, maximum corners).
    pub fn grow(&mut self, min: Vec3A, max: Vec3A) {
        if !(min.is_finite() && max.is_finite()) {
            return;
        }
        self.bounds = Some(match self.bounds {
            Some((old_min, old_max)) => (old_min.min(min), old_max.max(max)),
            None => (min, max),
        });
    }
}

/// The world-space box around an oriented box.
fn world_box(aabb: &Aabb, transform: &GlobalTransform) -> (Vec3A, Vec3A) {
    let affine = transform.affine();
    let center = affine.transform_point3a(aabb.center);
    let matrix = affine.matrix3;
    let half = matrix.x_axis.abs() * aabb.half_extents.x
        + matrix.y_axis.abs() * aabb.half_extents.y
        + matrix.z_axis.abs() * aabb.half_extents.z;
    (center - half, center + half)
}

/// Grows [`ShadowSceneBounds`] by the meshes whose bounds or transform
/// changed (every mesh in its first run and after a reset).
pub fn update_shadow_scene_bounds(
    mut scene_bounds: ResMut<ShadowSceneBounds>,
    changed: Query<
        (&Aabb, &GlobalTransform),
        (With<Mesh3d>, Or<(Changed<Aabb>, Changed<GlobalTransform>)>),
    >,
    all: Query<(&Aabb, &GlobalTransform), With<Mesh3d>>,
    mut started: Local<bool>,
) {
    let full = !*started || scene_bounds.full;
    *started = true;
    if full {
        scene_bounds.full = false;
        for (aabb, transform) in &all {
            let (min, max) = world_box(aabb, transform);
            scene_bounds.grow(min, max);
        }
    } else {
        for (aabb, transform) in &changed {
            let (min, max) = world_box(aabb, transform);
            scene_bounds.grow(min, max);
        }
    }
}

/// The distance from a camera to the farthest point its cascade slice from
/// `near` to `far` (positive distances) can reach in any orientation: its far
/// corners. A function of the projection only, so the camera's rotation
/// can't change it; rounded up to whole units.
pub(crate) fn far_coverage_radius(projection: &Projection, near: f32, far: f32) -> f32 {
    projection
        .get_frustum_corners(-near, -far)
        .iter()
        .map(|corner| corner.length())
        .fold(0.0, f32::max)
        .ceil()
}

/// What decides a far projection besides the camera's position and the
/// scene's bounds: when any of it changes, the projection is built anew.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FarProjectionKey {
    rotation: [u32; 4],
    map_size: usize,
    radius: u32,
    guard: u32,
    filter_texels: u32,
}

/// What a far projection is built from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FarProjectionInputs {
    /// The light's rotation (only its direction counts).
    pub rotation: Quat,
    /// The camera's world position.
    pub camera: Vec3A,
    /// [`far_coverage_radius`].
    pub radius: f32,
    pub map_size: usize,
    pub guard: f32,
    pub filter_texels: f32,
    /// The scene's bounds ([`ShadowSceneBounds`]).
    pub scene_bounds: Option<(Vec3A, Vec3A)>,
}

/// A camera's retained far projection.
#[derive(Clone, Debug)]
pub(crate) struct RetainedFarProjection {
    key: FarProjectionKey,
    /// The side of the covered square, whole units.
    diameter: f32,
    texel: f32,
    /// How far (in light space, along each axis) the camera may be from
    /// `center` with its whole sphere still covered.
    guard: f32,
    /// Light-space x and y of the center: whole texels.
    center: Vec2,
    /// The light-space depth interval (low, high; high faces the light).
    depth: (f32, f32),
    pub(crate) cascade: Cascade,
    /// Times built (see [`StableFarCascade::rebuilds`]).
    rebuilds: u64,
}

/// The light-space depth interval a far projection needs: the scene's
/// bounds, or without any the camera's sphere.
fn needed_depth(light_from_world: &Mat4, inputs: &FarProjectionInputs) -> (f32, f32) {
    let (lo, hi) = match inputs.scene_bounds {
        Some((min, max)) => {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for corner in 0..8 {
                let point = Vec3A::new(
                    if corner & 1 == 0 { min.x } else { max.x },
                    if corner & 2 == 0 { min.y } else { max.y },
                    if corner & 4 == 0 { min.z } else { max.z },
                );
                let z = light_from_world.transform_point3a(point).z;
                lo = lo.min(z);
                hi = hi.max(z);
            }
            (lo, hi)
        }
        None => {
            let z = light_from_world.transform_point3a(inputs.camera).z;
            (z - inputs.radius, z + inputs.radius)
        }
    };
    // At least a unit deep.
    let mid = 0.5 * (lo + hi);
    (lo.min(mid - 0.5), hi.max(mid + 0.5))
}

/// The retained interval for a needed one: rounded outward to power-of-two
/// bins (about an eighth of its depth), padded by a bin at both ends.
fn retained_depth((lo, hi): (f32, f32)) -> (f32, f32) {
    let bin = ops::exp2(ops::log2(((hi - lo) / 8.0).max(1.0)).ceil());
    (
        ((lo / bin).floor() - 1.0) * bin,
        ((hi / bin).ceil() + 1.0) * bin,
    )
}

/// A cascade's matrices from the light's rotation, the light-space point at
/// the center of its near plane, its square's side and depth.
fn cascade_from_parts(
    world_from_light: Mat4,
    near_plane_center: Vec3A,
    diameter: f32,
    depth: f32,
    texel: f32,
) -> Cascade {
    // As Bevy's `calculate_cascade`.
    let world_from_light_transpose = world_from_light.transpose();
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-near_plane_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_from_light * near_plane_center.extend(1.0),
    );
    let r = depth.recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );
    Cascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world: clip_from_cascade * cascade_from_world,
        texel_size: texel,
    }
}

impl RetainedFarProjection {
    /// Updates a camera's far projection for this frame: kept bit for bit
    /// while the key holds, the camera stays within the guard and the scene
    /// within the depth interval; built anew otherwise. Returns whether it
    /// changed.
    pub(crate) fn update(
        retained: &mut Option<Self>,
        inputs: &FarProjectionInputs,
        recenter: bool,
    ) -> bool {
        let key = FarProjectionKey {
            rotation: inputs.rotation.to_array().map(f32::to_bits),
            map_size: inputs.map_size,
            radius: inputs.radius.to_bits(),
            guard: inputs.guard.to_bits(),
            filter_texels: inputs.filter_texels.to_bits(),
        };
        let world_from_light = Mat4::from_quat(inputs.rotation);
        let light_from_world = world_from_light.transpose();
        let camera = light_from_world.transform_point3a(inputs.camera);
        let camera = Vec2::new(camera.x, camera.y);
        let needed = needed_depth(&light_from_world, inputs);

        let previous = retained.as_ref().filter(|old| old.key == key);
        let size = inputs.map_size as f32;
        let (diameter, texel, guard) = match previous {
            Some(old) => (old.diameter, old.texel, old.guard),
            None => {
                // The square holds the sphere, the guard and the filter's
                // texels at both sides; a whole number of units over a power
                // of two texels is exact.
                let usable = (1.0 - 2.0 * inputs.filter_texels / size).max(0.5);
                let diameter = (2.0 * inputs.radius * (1.0 + inputs.guard) / usable).ceil();
                let texel = diameter / size;
                let guard = 0.5 * diameter - inputs.radius - inputs.filter_texels * texel;
                (diameter, texel, guard.max(0.0))
            }
        };
        let center = match previous {
            Some(old) if (camera - old.center).abs().max_element() <= guard || !recenter => {
                old.center
            }
            _ => (camera / texel).round() * texel,
        };
        let depth = match previous {
            Some(old) if old.depth.0 <= needed.0 && needed.1 <= old.depth.1 => old.depth,
            _ => retained_depth(needed),
        };
        if let Some(old) = previous
            && old.center == center
            && old.depth == depth
        {
            return false;
        }
        let rebuilds = retained.as_ref().map_or(0, |old| old.rebuilds) + 1;
        let cascade = cascade_from_parts(
            world_from_light,
            Vec3A::new(center.x, center.y, depth.1),
            diameter,
            depth.1 - depth.0,
            texel,
        );
        *retained = Some(RetainedFarProjection {
            key,
            diameter,
            texel,
            guard,
            center,
            depth,
            cascade,
            rebuilds,
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_camera::{primitives::Frustum, PerspectiveProjection};
    use bevy_math::{Affine3A, EulerRot, Vec3, Vec3Swizzles, Vec4Swizzles};

    const SIZE: usize = 2048;

    fn projection() -> Projection {
        Projection::Perspective(PerspectiveProjection {
            fov: 50f32.to_radians(),
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 5000.0,
            ..Default::default()
        })
    }

    /// The far cascade of a three-cascade sun to 3000 m, as MPS's Crater:
    /// its slice from the overlapped near bound to 3000 m.
    const NEAR: f32 = 0.8 * 245.0;
    const FAR: f32 = 3000.0;

    fn sun(elevation_degrees: f32) -> Quat {
        Quat::from_euler(EulerRot::YXZ, 0.7, -elevation_degrees.to_radians(), 0.0)
    }

    fn inputs(
        rotation: Quat,
        camera: Vec3A,
        bounds: Option<(Vec3A, Vec3A)>,
    ) -> FarProjectionInputs {
        FarProjectionInputs {
            rotation,
            camera,
            radius: far_coverage_radius(&projection(), NEAR, FAR),
            map_size: SIZE,
            guard: 0.125,
            filter_texels: 8.0,
            scene_bounds: bounds,
        }
    }

    fn crater() -> Option<(Vec3A, Vec3A)> {
        Some((
            Vec3A::new(-4000.0, -50.0, -4000.0),
            Vec3A::new(4000.0, 600.0, 4000.0),
        ))
    }

    fn bits(cascade: &Cascade) -> [u32; 16] {
        cascade.clip_from_world.to_cols_array().map(f32::to_bits)
    }

    fn clip(cascade: &Cascade, point: Vec3A) -> Vec3A {
        let clip = cascade.clip_from_world * point.extend(1.0);
        Vec3A::from(clip.xyz() / clip.w)
    }

    /// Cheap deterministic numbers in [0, 1).
    fn numbers(seed: u32) -> impl Iterator<Item = f32> {
        let mut state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
        core::iter::from_fn(move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            Some((state >> 8) as f32 / (1u32 << 24) as f32)
        })
    }

    #[test]
    fn rotation_never_moves_the_projection_and_the_slice_stays_covered() {
        let rotation = sun(35.0);
        let camera = Vec3A::new(120.0, 30.0, -80.0);
        let mut retained = None;
        assert!(RetainedFarProjection::update(
            &mut retained,
            &inputs(rotation, camera, crater()),
            true
        ));
        let first = bits(&retained.as_ref().unwrap().cascade);
        let margin = 2.0 * 8.0 / SIZE as f32;
        let mut random = numbers(1);
        for _ in 0..500 {
            let (yaw, pitch, roll) = (
                random.next().unwrap() * 6.3,
                (random.next().unwrap() - 0.5) * 3.1,
                (random.next().unwrap() - 0.5) * 0.5,
            );
            // The camera's rotation doesn't enter the inputs at all: the
            // projection is the same bits.
            assert!(!RetainedFarProjection::update(
                &mut retained,
                &inputs(rotation, camera, crater()),
                true
            ));
            let cascade = &retained.as_ref().unwrap().cascade;
            assert_eq!(bits(cascade), first);
            // Every corner of the far slice, in this orientation, is inside
            // the map with the filter's texels to spare, and inside the depth
            // interval.
            let world_from_view = Affine3A::from_rotation_translation(
                Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll),
                camera.into(),
            );
            for corner in projection().get_frustum_corners(-NEAR, -FAR) {
                let point = clip(cascade, world_from_view.transform_point3a(corner));
                assert!(
                    point.x.abs() <= 1.0 - margin && point.y.abs() <= 1.0 - margin,
                    "{point:?}"
                );
            }
        }
    }

    #[test]
    fn only_a_translation_beyond_the_guard_recenters() {
        let rotation = sun(35.0);
        let start = Vec3A::new(10.0, 2.0, 10.0);
        let mut retained = None;
        RetainedFarProjection::update(&mut retained, &inputs(rotation, start, crater()), true);
        let first = retained.clone().unwrap();
        // The guard holds at least its share of the radius.
        assert!(first.guard >= 0.125 * inputs(rotation, start, crater()).radius);
        // The light's axes in the world: moving along the light's z (towards
        // the sun) changes nothing either.
        let light_x = Vec3A::from(rotation * Vec3::X);
        let light_y = Vec3A::from(rotation * Vec3::Y);
        let light_z = Vec3A::from(rotation * Vec3::Z);
        for step in [
            light_x * (first.guard * 0.99),
            light_y * (-first.guard * 0.99),
            light_z * 300.0,
            (light_x + light_y) * (first.guard * 0.7),
        ] {
            let mut moved = retained.clone();
            assert!(!RetainedFarProjection::update(
                &mut moved,
                &inputs(rotation, start + step, crater()),
                true
            ));
            assert_eq!(bits(&moved.unwrap().cascade), bits(&first.cascade));
        }
        // Beyond the guard: re-centered on whole texels, same texel size and
        // depth, and the texel lattice is the same in the world (a point's
        // texel coordinate moves by whole texels).
        let mut moved = retained.clone();
        assert!(RetainedFarProjection::update(
            &mut moved,
            &inputs(rotation, start + light_x * (first.guard * 1.5), crater()),
            true
        ));
        let moved = moved.unwrap();
        assert_eq!(moved.texel, first.texel);
        assert_eq!(moved.depth, first.depth);
        for center in [first.center, moved.center] {
            let texels = center / first.texel;
            assert_eq!(texels, texels.round());
        }
        let mut random = numbers(2);
        for _ in 0..100 {
            let point = Vec3A::new(
                (random.next().unwrap() - 0.5) * 2000.0,
                random.next().unwrap() * 200.0,
                (random.next().unwrap() - 0.5) * 2000.0,
            );
            let texel = |cascade: &Cascade| (clip(cascade, point).xy() * 0.5 + 0.5) * SIZE as f32;
            let shift = texel(&moved.cascade) - texel(&first.cascade);
            assert!(
                (shift - shift.round()).abs().max_element() < 2e-3,
                "{shift:?}"
            );
        }
        assert_eq!(moved.rebuilds, 2);
    }

    #[test]
    fn depth_covers_the_scene_and_holds_until_it_grows_past() {
        let rotation = sun(20.0);
        let camera = Vec3A::new(0.0, 5.0, 0.0);
        let mut retained = None;
        RetainedFarProjection::update(&mut retained, &inputs(rotation, camera, crater()), true);
        let first = retained.clone().unwrap();
        // Every corner of the scene maps into [0, 1].
        let (min, max) = crater().unwrap();
        for corner in 0..8 {
            let point = Vec3A::new(
                if corner & 1 == 0 { min.x } else { max.x },
                if corner & 2 == 0 { min.y } else { max.y },
                if corner & 4 == 0 { min.z } else { max.z },
            );
            let depth = clip(&first.cascade, point).z;
            assert!((0.0..=1.0).contains(&depth), "{depth}");
        }
        // Smaller bounds and a little growth inside the padding keep it.
        let smaller = Some((min * 0.5, max * 0.5));
        let grown = Some((min - Vec3A::splat(1.0), max + Vec3A::splat(1.0)));
        for bounds in [smaller, grown, None] {
            let mut kept = retained.clone();
            assert!(!RetainedFarProjection::update(
                &mut kept,
                &inputs(rotation, camera, bounds),
                true
            ));
            assert_eq!(bits(&kept.unwrap().cascade), bits(&first.cascade));
        }
        // Growth past the interval: a new interval that covers it.
        let tower = Some((min, Vec3A::new(max.x, 50_000.0, max.z)));
        let mut grown = retained.clone();
        assert!(RetainedFarProjection::update(
            &mut grown,
            &inputs(rotation, camera, tower),
            true
        ));
        let grown = grown.unwrap();
        assert_eq!((grown.center, grown.texel), (first.center, first.texel));
        assert!(grown.depth.0 <= first.depth.0 && grown.depth.1 > first.depth.1);
        // The bins are powers of two and the ends whole bins.
        let bin = ((first.depth.1 - first.depth.0) / 8.0).max(1.0);
        assert!(bin >= 1.0);
        assert_eq!(first.depth.0, first.depth.0.round());
    }

    #[test]
    fn a_new_sun_direction_or_map_size_rebuilds() {
        let camera = Vec3A::new(0.0, 5.0, 0.0);
        let mut retained = None;
        RetainedFarProjection::update(&mut retained, &inputs(sun(35.0), camera, crater()), true);
        let first = bits(&retained.as_ref().unwrap().cascade);
        assert!(RetainedFarProjection::update(
            &mut retained,
            &inputs(sun(35.001), camera, crater()),
            true
        ));
        assert_ne!(bits(&retained.as_ref().unwrap().cascade), first);
        let mut bigger = inputs(sun(35.001), camera, crater());
        bigger.map_size = 4096;
        assert!(RetainedFarProjection::update(&mut retained, &bigger, true));
        let retained = retained.unwrap();
        assert_eq!(retained.texel, retained.diameter / 4096.0);
    }

    #[test]
    fn a_low_sun_keeps_casters_far_upstream() {
        // A sun 3 degrees above the horizon: a ridge 20 km upstream of a
        // covered receiver shadows it. It shares the receiver's light-space x
        // and y, so the cascade's frustum (with its near plane pushed to
        // infinity, as the shadow views and the lights' visibility test it)
        // keeps it, and its depth clamps at the near end.
        let rotation = sun(3.0);
        let camera = Vec3A::new(0.0, 2.0, 0.0);
        let mut retained = None;
        RetainedFarProjection::update(&mut retained, &inputs(rotation, camera, crater()), true);
        let cascade = retained.unwrap().cascade;
        let towards_sun = Vec3A::from(rotation * Vec3::Z);
        let receiver = Vec3A::new(1500.0, 0.0, -1200.0);
        assert!(clip(&cascade, receiver).xy().abs().max_element() < 1.0);
        let ridge = receiver + towards_sun * 20_000.0;
        let point = clip(&cascade, ridge);
        assert!(point.xy().abs().max_element() < 1.0);
        assert!(point.z > 1.0);
        let frustum = Frustum(bevy_shape::ViewFrustum::from_clip_from_world(
            &cascade.clip_from_world,
        ));
        let caster = Aabb {
            center: ridge,
            half_extents: Vec3A::splat(5.0),
        };
        assert!(frustum.intersects_obb(&caster, &Affine3A::IDENTITY, false, true));
        // Not a degenerate test: a caster beside the coverage is out.
        let beside = Aabb {
            center: ridge + Vec3A::from(rotation * Vec3::X) * 10_000.0,
            half_extents: Vec3A::splat(5.0),
        };
        assert!(!frustum.intersects_obb(&beside, &Affine3A::IDENTITY, false, true));
    }

    #[test]
    fn views_take_casters_by_representation() {
        use ShadowRepresentation::*;
        for selects in [false, true] {
            assert!(
                view_takes(selects, None),
                "unmarked casters cast everywhere"
            );
        }
        assert!(view_takes(false, Some(Detail)));
        assert!(!view_takes(true, Some(Detail)));
        assert!(!view_takes(false, Some(FarCascadeStandIn)));
        assert!(view_takes(true, Some(FarCascadeStandIn)));
    }

    #[test]
    fn a_stand_in_is_never_camera_visible() {
        let mut world = World::new();
        let stand_in = world.spawn(ShadowRepresentation::FarCascadeStandIn).id();
        let detail = world.spawn(ShadowRepresentation::Detail).id();
        world.flush();
        assert!(world.get::<NotCameraVisible>(stand_in).is_some());
        assert!(world.get::<NotCameraVisible>(detail).is_none());
        // Made a stand-in later.
        world
            .entity_mut(detail)
            .insert(ShadowRepresentation::FarCascadeStandIn);
        world.flush();
        assert!(world.get::<NotCameraVisible>(detail).is_some());
    }

    #[test]
    fn scene_bounds_grow_and_reset() {
        let mut bounds = ShadowSceneBounds::default();
        assert!(bounds.bounds().is_none());
        bounds.grow(Vec3A::splat(-1.0), Vec3A::splat(1.0));
        bounds.grow(Vec3A::new(0.0, 0.0, 0.0), Vec3A::new(5.0, 0.5, 0.5));
        bounds.grow(Vec3A::splat(f32::NAN), Vec3A::splat(9.0));
        assert_eq!(
            bounds.bounds(),
            Some((Vec3A::splat(-1.0), Vec3A::new(5.0, 1.0, 1.0)))
        );
        bounds.reset();
        assert!(bounds.bounds().is_none());
        // A rotated box's world bounds hold its corners.
        let transform = GlobalTransform::from(bevy_transform::components::Transform {
            translation: Vec3::new(3.0, 0.0, 0.0),
            rotation: Quat::from_rotation_y(0.6),
            scale: Vec3::new(1.0, 2.0, 3.0),
        });
        let aabb = Aabb {
            center: Vec3A::new(0.1, 0.2, 0.3),
            half_extents: Vec3A::new(1.0, 0.5, 0.25),
        };
        let (min, max) = world_box(&aabb, &transform);
        for corner in 0..8 {
            let sign = |bit: u32| if corner & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let local = aabb.center + aabb.half_extents * Vec3A::new(sign(0), sign(1), sign(2));
            let world = transform.affine().transform_point3a(local);
            assert!(world.cmpge(min - 1e-4).all() && world.cmple(max + 1e-4).all());
        }
    }
}
