//! Illustrates spot lights, and caching the shadows of casters that stand
//! still while moving ones are drawn over them.
//!
//! `--oracle` runs a scripted scene instead of the interactive one and checks
//! every cached shadow map against an independent full redraw, texel by
//! texel; see the `oracle` module.

use std::f32::consts::*;

use bevy::{
    camera::Hdr,
    color::palettes::basic::{MAROON, RED},
    light::{NotShadowCaster, ShadowCacheFeedback, ShadowCacheSettings},
    math::ops,
    prelude::*,
};
use chacha20::ChaCha8Rng;
use rand::{RngExt, SeedableRng};

const INSTRUCTIONS: &str = "\
Controls
--------
Horizontal Movement: WASD
Vertical Movement: Space and Shift
Rotate Camera: Left and Right Arrows
Shadow Cache On and Off (the lights sway while it's off): K";

fn main() -> AppExit {
    // Measuring scaffolding: a level with many lights and casters.
    if std::env::args().any(|arg| arg == "--stress") {
        return stress::run(std::env::args().any(|arg| arg == "--oracle"));
    }
    let oracle = std::env::args().any(|arg| arg == "--oracle");
    let mut app = App::new();
    app.insert_resource(GlobalAmbientLight {
        brightness: 20.0,
        ..default()
    })
    .insert_resource(ShadowCacheSettings {
        enabled: true,
        ..default()
    })
    .add_plugins(DefaultPlugins)
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            toggle_shadow_cache,
            // Not under the oracle, whose runs with the cache off must show
            // the same scene.
            light_sway
                .run_if(move |settings: Res<ShadowCacheSettings>| !oracle && !settings.enabled),
            movement,
            rotation,
            orbit,
        )
            .chain(),
    )
    .add_systems(Update, report_shadow_cache);
    // Test scaffolding: checks the cached shadow maps against full redraws.
    if oracle {
        app.add_plugins(oracle::OraclePlugin { scripted: true });
    }
    app.run()
}

#[derive(Component)]
struct Movable;

/// A shadow caster that moves by itself: its shadow is drawn every frame.
#[derive(Component)]
struct Orbiting;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(100.0, 100.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Movable,
    ));

    // cubes

    // We're seeding the PRNG here to make this example deterministic for testing purposes.
    // This isn't strictly required in practical use unless you need your app to be deterministic.
    let mut rng = ChaCha8Rng::seed_from_u64(19878367467713);
    let cube_mesh = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
    let blue = materials.add(Color::srgb_u8(124, 144, 255));
    let orbiting_mesh = cube_mesh.clone();

    commands.spawn_batch(
        std::iter::repeat_with(move || {
            let x = rng.random_range(-5.0..5.0);
            let y = rng.random_range(0.0..3.0);
            let z = rng.random_range(-5.0..5.0);

            (
                Mesh3d(cube_mesh.clone()),
                MeshMaterial3d(blue.clone()),
                Transform::from_xyz(x, y, z),
                Movable,
            )
        })
        .take(40),
    );

    // A cube that moves: its shadow is drawn every frame.
    commands.spawn((
        Mesh3d(orbiting_mesh),
        MeshMaterial3d(materials.add(Color::srgb_u8(255, 200, 80))),
        Transform::from_xyz(0.0, 0.75, 0.0),
        Movable,
        Orbiting,
    ));

    let sphere_mesh = meshes.add(Sphere::new(0.05).mesh().uv(32, 18));
    let sphere_mesh_direction = meshes.add(Sphere::new(0.1).mesh().uv(32, 18));
    let red_emissive = materials.add(StandardMaterial {
        base_color: RED.into(),
        emissive: LinearRgba::new(1.0, 0.0, 0.0, 0.0),
        ..default()
    });
    let maroon_emissive = materials.add(StandardMaterial {
        base_color: MAROON.into(),
        emissive: LinearRgba::new(0.369, 0.0, 0.0, 0.0),
        ..default()
    });

    for x in 0..4 {
        for z in 0..4 {
            let x = x as f32 - 2.0;
            let z = z as f32 - 2.0;
            // red spot_light
            commands.spawn((
                SpotLight {
                    intensity: 40_000.0, // lumens
                    color: Color::WHITE,
                    shadow_maps_enabled: true,
                    inner_angle: PI / 4.0 * 0.85,
                    outer_angle: PI / 4.0,
                    ..default()
                },
                Transform::from_xyz(1.0 + x, 2.0, z)
                    .looking_at(Vec3::new(1.0 + x, 0.0, z), Vec3::X),
                children![
                    (
                        // It crosses its light's near plane: while it is
                        // live (when its light moved) it restores the whole
                        // map every frame.
                        Mesh3d(sphere_mesh.clone()),
                        MeshMaterial3d(red_emissive.clone()),
                    ),
                    (
                        Mesh3d(sphere_mesh_direction.clone()),
                        MeshMaterial3d(maroon_emissive.clone()),
                        Transform::from_translation(Vec3::Z * -0.1),
                        NotShadowCaster,
                    )
                ],
            ));
        }
    }

    // camera
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Transform::from_xyz(-4.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Text::new(INSTRUCTIONS),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn toggle_shadow_cache(
    input: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<ShadowCacheSettings>,
) {
    if input.just_pressed(KeyCode::KeyK) {
        settings.enabled = !settings.enabled;
    }
}

fn light_sway(time: Res<Time>, mut query: Query<(&mut Transform, &mut SpotLight)>) {
    for (mut transform, mut angles) in query.iter_mut() {
        transform.rotation = Quat::from_euler(
            EulerRot::XYZ,
            -FRAC_PI_2 + ops::sin(time.elapsed_secs() * 0.67 * 3.0) * 0.5,
            ops::sin(time.elapsed_secs() * 3.0) * 0.5,
            0.0,
        );
        let angle = (ops::sin(time.elapsed_secs() * 1.2) + 1.0) * (FRAC_PI_4 - 0.1);
        angles.inner_angle = angle * 0.8;
        angles.outer_angle = angle;
    }
}

fn movement(
    input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Movable>>,
) {
    // Calculate translation to move the cubes and ground plane
    let mut translation = Vec3::ZERO;

    // Horizontal forward and backward movement
    if input.pressed(KeyCode::KeyW) {
        translation.z += 1.0;
    } else if input.pressed(KeyCode::KeyS) {
        translation.z -= 1.0;
    }

    // Horizontal left and right movement
    if input.pressed(KeyCode::KeyA) {
        translation.x += 1.0;
    } else if input.pressed(KeyCode::KeyD) {
        translation.x -= 1.0;
    }

    // Vertical movement
    if input.pressed(KeyCode::ShiftLeft) {
        translation.y += 1.0;
    } else if input.pressed(KeyCode::Space) {
        translation.y -= 1.0;
    }

    translation *= 2.0 * time.delta_secs();
    if translation == Vec3::ZERO {
        return;
    }

    // Apply translation
    for mut transform in &mut query {
        transform.translation += translation;
    }
}

fn rotation(
    mut transform: Single<&mut Transform, With<Camera>>,
    input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let delta = time.delta_secs();

    if input.pressed(KeyCode::ArrowLeft) {
        transform.rotate_around(Vec3::ZERO, Quat::from_rotation_y(delta));
    } else if input.pressed(KeyCode::ArrowRight) {
        transform.rotate_around(Vec3::ZERO, Quat::from_rotation_y(-delta));
    }
}

fn orbit(time: Res<Time>, mut query: Query<&mut Transform, With<Orbiting>>) {
    let angle = time.elapsed_secs() * 0.8;
    for mut transform in &mut query {
        transform.translation.x = 3.0 * ops::cos(angle);
        transform.translation.z = 3.0 * ops::sin(angle);
    }
}

/// Logs what the cached lights did, every two seconds.
fn report_shadow_cache(time: Res<Time>, feedback: Res<ShadowCacheFeedback>, mut next: Local<f32>) {
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 2.0;
    let mut states = std::collections::BTreeMap::new();
    let mut restored = 0;
    for status in feedback.snapshot().values() {
        *states.entry(format!("{:?}", status.state)).or_insert(0) += 1;
        restored += status.restored_texels;
    }
    info!("shadow cache: {states:?}, {restored} texels restored");
}

/// Test scaffolding for the spot light shadow cache (`--oracle`).
///
/// Bevy's `ShadowCacheAuditPlugin` gives every shadowed spot light a twin
/// drawn from scratch every frame and compares their shadow maps texel by
/// texel after each frame's passes. A scripted scene exercises the cache (see
/// [`scenario`]), and the script checks, frame by frame, which casters are
/// settled ([`Expect`]) and that rewriting unchanged transforms promotes
/// nothing. The run passes if no checked frame differs and every check holds,
/// and exits with an error otherwise.
///
/// Environment: `SPOTLIGHT_ORACLE_BUDGET_MIB` sets the static layers' budget
/// (0: every light falls back to full redraws);
/// `SPOTLIGHT_ORACLE_RESTORE_NOT_READY=1` keeps the restore's blit counted as
/// not compiled, so every light falls back as in the first frames;
/// `SPOTLIGHT_ORACLE_INVALIDATE_EVERY=n` redraws every static layer every n
/// frames; `SPOTLIGHT_ORACLE_READINESS_DELAY=n` holds every mesh's shadow
/// pipeline back for its first n frames (4; 0 for none);
/// `SPOTLIGHT_ORACLE_NO_CPU_CULLING=1` culls every mesh on the GPU only;
/// `SPOTLIGHT_ORACLE_HASHES=file` runs without twins and writes a hash of
/// every shadowed spot light's map per frame to `file`, for a comparison with
/// a run with `SPOTLIGHT_ORACLE_CACHE=0` (the cache off: every map drawn
/// from scratch); `SPOTLIGHT_ORACLE_FROM` and `SPOTLIGHT_ORACLE_EXIT_AT` move
/// the first checked frame and the end of the run (for quick loops over the
/// start-up); `SHADOW_CACHE_FAULT` injects a fault that the oracle must catch
/// (`skip_restore`, `drop_written`, `drop_one_tile`,
/// `miss_static_invalidation`, `miss_settle_invalidation`, `late_marker`,
/// `stale_ack`, `wrong_layer`, `acknowledge_early`, `skip_reentry_restore`).
mod oracle {
    use super::*;
    use bevy::{
        camera::visibility::{NoCpuCulling, NoFrustumCulling, RenderLayers, VisibilityRange},
        diagnostic::FrameCount,
        light::{
            DirectionalLightShadowMap, InvalidateShadowCache, ShadowCacheCounters,
            ShadowCacheFaults, ShadowCacheLoading, ShadowCacheState, ShadowCasterLive,
            StaticShadowCaster,
        },
        mesh::Indices,
        pbr::{
            ShadowCacheAuditPlugin, ShadowCacheAuditResults, ShadowCacheAuditSettings,
            ShadowCacheAuditStatus,
        },
        platform::collections::HashMap,
        time::TimeUpdateStrategy,
        world_serialization::{WorldAsset, WorldInstanceReady},
    };
    use std::{io::Write, time::Duration};

    /// The first and last frames whose shadow maps are compared.
    const CHECK_FROM: u32 = 90;
    const CHECK_TO: u32 = 950;
    /// The frame the run ends in (read-backs arrive a few frames late).
    const EXIT_AT: u32 = 970;
    /// The frame the scene counts as loaded: casters that appear later wait
    /// the settle time.
    const LOADED_AT: u32 = 60;
    /// One frame of simulation time.
    const FRAME: Duration = Duration::from_nanos(33_333_333);
    /// The settle time, in frames.
    const SETTLE: u32 = 10;

    /// `SPOTLIGHT_ORACLE_EXIT_AT` ends the run earlier (for quick loops).
    fn exit_at() -> u32 {
        env("SPOTLIGHT_ORACLE_EXIT_AT").unwrap_or(EXIT_AT)
    }

    fn check_to() -> u32 {
        CHECK_TO.min(exit_at().saturating_sub(20))
    }

    /// `SPOTLIGHT_ORACLE_FROM` moves the first checked frame.
    fn check_from() -> u32 {
        env("SPOTLIGHT_ORACLE_FROM").unwrap_or(CHECK_FROM)
    }

    fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
    }

    fn hashes_file() -> Option<String> {
        std::env::var("SPOTLIGHT_ORACLE_HASHES").ok()
    }

    fn cache_on() -> bool {
        std::env::var("SPOTLIGHT_ORACLE_CACHE").as_deref() != Ok("0")
    }

    /// The oracle, on the scripted scene or on `--stress`'s.
    pub struct OraclePlugin {
        /// Runs the script (see [`scenario`]); without it the scene's own
        /// movers move.
        pub scripted: bool,
    }

    impl Plugin for OraclePlugin {
        fn build(&self, app: &mut App) {
            let budget_mib: u64 = env("SPOTLIGHT_ORACLE_BUDGET_MIB").unwrap_or(256);
            let hashes = hashes_file().is_some();
            app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
                .insert_resource(faults_from_env())
                .insert_resource(ShadowCacheSettings {
                    enabled: cache_on(),
                    budget_bytes: budget_mib << 20,
                    settle_time: FRAME * SETTLE,
                })
                .insert_resource(ShadowCacheAuditSettings {
                    enabled: true,
                    fresh_redraws: !hashes,
                    hashes,
                })
                .init_resource::<Script>()
                .add_plugins(ShadowCacheAuditPlugin)
                // Spot lights stay extracted out of view (as an app that
                // keeps its shadow map array from resizing does), so a cached
                // light that lights no view is planned, not dropped.
                .add_observer(|add: On<Add<SpotLight>>, mut commands: Commands| {
                    commands.entity(add.entity).insert(NoFrustumCulling);
                });
            if std::env::var("SPOTLIGHT_ORACLE_NO_CPU_CULLING").as_deref() == Ok("1") {
                app.add_observer(|add: On<Add<Mesh3d>>, mut commands: Commands| {
                    commands.entity(add.entity).insert(NoCpuCulling);
                });
            }
            if self.scripted {
                app.add_systems(
                    Startup,
                    (
                        list_lights.after(setup),
                        spawn_script_casters.after(list_lights),
                        warm_up_gltf.after(spawn_script_casters),
                    ),
                )
                .add_systems(Update, (run_script, move_movers, collect_results).chain())
                .add_systems(Last, check_expectations);
            } else {
                app.add_systems(Update, collect_results);
            }
        }
    }

    fn faults_from_env() -> ShadowCacheFaults {
        let fault = std::env::var("SHADOW_CACHE_FAULT").unwrap_or_default();
        let faults = ShadowCacheFaults {
            skip_restore: fault == "skip_restore",
            drop_written: fault == "drop_written",
            drop_one_tile: fault == "drop_one_tile",
            miss_static_invalidation: fault == "miss_static_invalidation",
            miss_settle_invalidation: fault == "miss_settle_invalidation",
            late_marker: fault == "late_marker",
            stale_ack: fault == "stale_ack",
            wrong_layer: fault == "wrong_layer",
            acknowledge_early: fault == "acknowledge_early",
            skip_reentry_restore: fault == "skip_reentry_restore",
            restore_not_ready: false,
            readiness_delay_frames: 0,
        };
        if !fault.is_empty() {
            assert_ne!(
                faults,
                ShadowCacheFaults::default(),
                "unknown fault {fault}"
            );
            info!("oracle: injected fault {fault}");
        }
        ShadowCacheFaults {
            restore_not_ready: std::env::var("SPOTLIGHT_ORACLE_RESTORE_NOT_READY").as_deref()
                == Ok("1"),
            // Longer than the settle time, so a new caster's lights rebuild
            // while its pipeline still waits.
            readiness_delay_frames: env("SPOTLIGHT_ORACLE_READINESS_DELAY").unwrap_or(15),
            ..faults
        }
    }

    /// A mover of the script.
    #[derive(Component)]
    struct Velocity(Vec3);

    /// Moves around a center.
    #[derive(Component)]
    struct Circling {
        center: Vec3,
        radius: f32,
        speed: f32,
    }

    /// An animation to start once a glTF scene is ready.
    #[derive(Component)]
    struct PlayAnimation {
        graph: Handle<AnimationGraph>,
        index: AnimationNodeIndex,
    }

    /// What the script spawned, and the results so far.
    #[derive(Resource, Default)]
    struct Script {
        lights: Vec<Entity>,
        spawned: Vec<(&'static str, Entity)>,
        /// Casters the script moves and checks, by name.
        named: HashMap<&'static str, Entity>,
        /// The audit's records of the checked frames.
        records: Vec<AuditedLightFrame>,
        /// Checks that failed: (frame, what).
        failed_checks: Vec<(u32, String)>,
        checks: u32,
        /// Promotions before the identical writes.
        promotions_before: u64,
        /// The counters before the short-lived spawns.
        counters_before: Option<ShadowCacheCounters>,
        /// The glTF assets the script spawns, loaded at the start and kept.
        assets: Vec<UntypedHandle>,
        /// The mesh and the material replaced under the same id.
        swap_mesh: Handle<Mesh>,
        swap_material: Handle<StandardMaterial>,
        /// With `SPOTLIGHT_ORACLE_HASHES`: (frame, light, hash, cached, what
        /// the cache did).
        hashes: Vec<(u32, Entity, String, bool, String)>,
    }

    /// What the oracle keeps of an audit record.
    struct AuditedLightFrame {
        frame: u32,
        light: Entity,
        status: ShadowCacheAuditStatus,
        differing: u32,
        /// The light-frames the audit expected in the frame.
        expected: u32,
    }

    /// What a frame of the script tests.
    pub fn scenario(frame: u32) -> &'static str {
        match frame {
            0..100 => "static scene, rigid mover",
            100..130 => "skinned Fox",
            130..160 => "morph targets",
            160..190 => "spawn burst",
            190..210 => "despawn burst",
            210..240 => "light switched off",
            240..260 => "light switched on",
            260..280 => "light moved",
            280..300 => "near-plane crossing",
            300..320 => "near-plane mover gone",
            320..340 => "last movers leaving",
            340..360 => "caster added",
            360..380 => "caster removed",
            380..420 => "lights renumbered",
            420..440 => "a settled caster moved once",
            440..455 => "new casters (their pipelines late), settling after the settle time",
            455..480 => {
                "moves during the rebuild that waits, before its ack; settling under another mover"
            }
            480..490 => "the last mover leaving",
            490..520 => "a light moved, a promotion and a settle in one frame",
            520..540 => "shadow map size changed, a promotion in the same frame",
            540..560 => "a light's render layers changed",
            560..580 => "size back, a child reparented",
            580..600 => "identical transform writes",
            600..690 => "micro-motion, slow drift, pauses shorter and longer than the settle time",
            690..750 => "forced live added and removed",
            750..800 => "debris settling",
            800..820 => "runtime building",
            820..845 => "short-lived spawns that never settle",
            845..850 => "movers circling under four lights",
            850..858 => "every light out of view, its movers moving",
            858..870 => "back in view, a settled caster moved in the same frame",
            870..881 => "a new caster settles, its light's rebuild waits for its pipeline",
            881..883 => "out of view during the unacknowledged rebuild",
            883..900 => "back in view during the unacknowledged rebuild",
            900..912 => "out of view while the movers leave",
            912..915 => "back in view after the movers left",
            915..930 => "a settled caster's mesh replaced under the same id, same bounds",
            _ => "a settled caster's material replaced under the same id",
        }
    }

    /// Frames at which a named caster must be settled (or live), checked
    /// after the frame's classification.
    struct Expect {
        frame: u32,
        name: &'static str,
        settled: bool,
    }

    fn expectations() -> Vec<Expect> {
        let mut expect = Vec::new();
        let mut at = |frame, name, settled| {
            expect.push(Expect {
                frame,
                name,
                settled,
            });
        };
        // Everything the script names settles in the frame after it appears.
        for name in [
            "door", "child", "drifter", "jitter_a", "jitter_b", "forced", "micro", "settler",
        ] {
            at(100, name, true);
        }
        // A promotion in its frame, a settle after the settle time.
        at(420, "door", false);
        at(428, "door", false);
        at(440, "door", true);
        // A spawn after loading waits the settle time; its light's rebuild
        // then waits for its pipeline (the readiness delay).
        at(441, "late", false);
        at(456, "late", true);
        at(453, "door", false);
        at(462, "door", false);
        at(476, "door", true);
        at(479, "settler", false);
        at(490, "child", false);
        at(492, "settler", true);
        at(520, "door", false);
        at(545, "door", true);
        at(571, "child", false);
        at(595, "child", true);
        // One bit of motion is motion; a slow drift never settles.
        at(600, "micro", false);
        at(625, "micro", true);
        for frame in [610, 620, 630, 639] {
            at(frame, "drifter", false);
        }
        at(660, "drifter", true);
        // Pauses shorter than the settle time never settle; longer ones do.
        for frame in (601..690).step_by(7) {
            at(frame, "jitter_a", false);
        }
        for start in (600..675).step_by(15) {
            at(start, "jitter_b", false);
            at(start + 14, "jitter_b", true);
        }
        // Forced live, and the settle time after it.
        at(690, "forced", false);
        at(705, "forced", false);
        at(718, "forced", true);
        at(720, "forced", false);
        at(735, "forced", false);
        at(748, "forced", true);
        at(805, "built", false);
        at(815, "built", true);
        at(859, "door", false);
        at(871, "late2", false);
        at(895, "late2", true);
        // An asset replaced under the same id is a change of its users.
        at(914, "swap_mesh", true);
        at(915, "swap_mesh", false);
        at(929, "swap_material", true);
        at(930, "swap_material", false);
        expect
    }

    fn list_lights(lights: Query<Entity, With<SpotLight>>, mut script: ResMut<Script>) {
        script.lights.extend(&lights);
        script.lights.sort();
    }

    /// Where a cube under a light stands.
    fn under(light: Vec3, offset: Vec3) -> Vec3 {
        Vec3::new(light.x, 0.3, light.z) + offset
    }

    /// The casters the script moves later, spawned with the scene (so they
    /// settle at the start), and a light behind the camera whose casters no
    /// camera sees.
    fn spawn_script_casters(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut script: ResMut<Script>,
        lights: Query<&Transform, With<SpotLight>>,
    ) {
        let position = |index: usize| lights.get(script.lights[index]).unwrap().translation;
        let material = MeshMaterial3d(materials.add(Color::srgb(0.4, 0.8, 0.4)));
        let cube = Mesh3d(meshes.add(Cuboid::from_length(0.25)));
        let mut named = Vec::new();
        for (name, light, offset) in [
            ("door", 1, Vec3::ZERO),
            ("drifter", 3, Vec3::ZERO),
            ("jitter_a", 4, Vec3::ZERO),
            ("jitter_b", 7, Vec3::ZERO),
            ("forced", 8, Vec3::ZERO),
            ("micro", 11, Vec3::ZERO),
            ("settler", 2, Vec3::new(-0.4, 0.0, 0.0)),
        ] {
            let entity = commands
                .spawn((
                    cube.clone(),
                    material.clone(),
                    Transform::from_translation(under(position(light), offset)),
                ))
                .id();
            named.push((name, entity));
        }
        // A child of a mesh-less parent, and a second parent to move it to.
        let parent = commands
            .spawn((
                Transform::from_translation(under(position(2), Vec3::ZERO)),
                Visibility::default(),
            ))
            .id();
        let parent2 = commands
            .spawn((
                Transform::from_translation(under(position(2), Vec3::new(0.4, 0.0, 0.0))),
                Visibility::default(),
            ))
            .id();
        let child = commands
            .spawn((
                cube.clone(),
                material.clone(),
                Transform::from_xyz(0.0, 0.1, 0.2),
                ChildOf(parent),
            ))
            .id();
        named.extend([("parent", parent), ("parent2", parent2), ("child", child)]);
        // A mesh and a material of their own, replaced under the same id later.
        let swap_positions = [position(10), position(9)];
        script.swap_mesh = meshes.add(Cuboid::from_length(0.3));
        script.swap_material = materials.add(Color::srgb(0.7, 0.7, 0.9));
        for (name, light, mesh, material) in [
            (
                "swap_mesh",
                swap_positions[0],
                script.swap_mesh.clone(),
                material.0.clone(),
            ),
            (
                "swap_material",
                swap_positions[1],
                cube.0.clone(),
                script.swap_material.clone(),
            ),
        ] {
            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::from_translation(under(light, Vec3::new(0.0, 0.0, 0.3))),
                ))
                .id();
            named.push((name, entity));
        }
        // Behind the camera, so its casters are found only by the light.
        let back = Vec3::new(-4.0, 2.0, 14.0);
        let back_light = commands
            .spawn((
                SpotLight {
                    intensity: 40_000.0,
                    shadow_maps_enabled: true,
                    inner_angle: PI / 4.0 * 0.85,
                    outer_angle: PI / 4.0,
                    ..default()
                },
                Transform::from_translation(back)
                    .looking_at(Vec3::new(back.x, 0.0, back.z), Vec3::X),
            ))
            .id();
        named.push(("back_light", back_light));
        script.named.extend(named);
    }

    /// Loads the script's glTF scenes at the start and shows them under the
    /// light behind the camera until frame 80, so their assets are loaded and
    /// their shadow pipelines compiled before the checked frames: a run with
    /// the cache off renders at another pace, and asynchronous loading or
    /// compiling would otherwise land in other frames (the twin-less runs
    /// compare two processes frame by frame).
    fn warm_up_gltf(
        mut commands: Commands,
        asset_server: Res<AssetServer>,
        mut graphs: ResMut<Assets<AnimationGraph>>,
        mut script: ResMut<Script>,
        lights: Query<&Transform, With<SpotLight>>,
    ) {
        let back = lights.get(script.named["back_light"]).unwrap().translation;
        for (path, offset) in [
            ("models/animated/Fox.glb", Vec3::new(-0.5, 0.0, 0.0)),
            (
                "models/animated/MorphStressTest.gltf",
                Vec3::new(0.5, 0.4, 0.0),
            ),
        ] {
            script.assets.push(
                asset_server
                    .load::<WorldAsset>(GltfAssetLabel::Scene(0).from_asset(path))
                    .untyped(),
            );
            let clip = GltfAssetLabel::Animation(if path.contains("Fox") { 2 } else { 0 });
            script.assets.push(
                asset_server
                    .load::<AnimationClip>(clip.from_asset(path))
                    .untyped(),
            );
            let scale = if path.contains("Fox") { 0.012 } else { 0.4 };
            let warm = spawn_gltf(
                &mut commands,
                &asset_server,
                &mut graphs,
                path,
                Transform::from_translation(Vec3::new(back.x, 0.0, back.z) + offset)
                    .with_scale(Vec3::splat(scale)),
            );
            script.spawned.push(("warm", warm));
        }
    }

    fn spawn_gltf(
        commands: &mut Commands,
        asset_server: &AssetServer,
        graphs: &mut Assets<AnimationGraph>,
        path: &'static str,
        transform: Transform,
    ) -> Entity {
        let (graph, index) = AnimationGraph::from_clip(asset_server.load(
            GltfAssetLabel::Animation(if path.contains("Fox") { 2 } else { 0 }).from_asset(path),
        ));
        commands
            .spawn((
                PlayAnimation {
                    graph: graphs.add(graph),
                    index,
                },
                WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(path))),
                transform,
            ))
            .observe(play_animation)
            .id()
    }

    fn play_animation(
        ready: On<WorldInstanceReady>,
        mut commands: Commands,
        children: Query<&Children>,
        animations: Query<&PlayAnimation>,
        mut players: Query<&mut AnimationPlayer>,
    ) {
        let Ok(animation) = animations.get(ready.entity) else {
            return;
        };
        for child in children.iter_descendants(ready.entity) {
            if let Ok(mut player) = players.get_mut(child) {
                player.play(animation.index).repeat();
                commands
                    .entity(child)
                    .insert(AnimationGraphHandle(animation.graph.clone()));
            }
        }
    }

    fn take(script: &mut Script, name: &str) -> Vec<Entity> {
        let (taken, kept) = script
            .spawned
            .drain(..)
            .partition::<Vec<_>, _>(|(spawned, _)| *spawned == name);
        script.spawned = kept;
        taken.into_iter().map(|(_, entity)| entity).collect()
    }

    type ScriptLights<'w, 's> = Query<
        'w,
        's,
        (
            &'static mut Transform,
            &'static mut Visibility,
            &'static SpotLight,
        ),
    >;
    type ScriptCasters<'w, 's> = Query<
        'w,
        's,
        &'static mut Transform,
        (Without<SpotLight>, Without<Movable>, Without<Camera3d>),
    >;
    type ScriptCamera<'w, 's> = Query<
        'w,
        's,
        &'static mut Transform,
        (With<Camera3d>, Without<SpotLight>, Without<Movable>),
    >;

    #[expect(clippy::too_many_arguments, reason = "a script")]
    fn run_script(
        mut commands: Commands,
        mut invalidations: MessageWriter<InvalidateShadowCache>,
        frame: Res<FrameCount>,
        mut script: ResMut<Script>,
        asset_server: Res<AssetServer>,
        mut graphs: ResMut<Assets<AnimationGraph>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut lights: ScriptLights,
        mut casters: ScriptCasters,
        mut movables: Query<&mut Transform, (With<Movable>, Without<SpotLight>)>,
        orbiting: Query<Entity, With<Orbiting>>,
        mut shadow_map: ResMut<DirectionalLightShadowMap>,
        (counters, mut loading, mut camera): (
            Res<ShadowCacheCounters>,
            ResMut<ShadowCacheLoading>,
            ScriptCamera,
        ),
    ) {
        let frame = frame.0;
        // `SPOTLIGHT_ORACLE_INVALIDATE_EVERY=n`: every cached light redraws
        // its static layer every n frames (a stress test of the transitions).
        if let Some(every) = env::<u32>("SPOTLIGHT_ORACLE_INVALIDATE_EVERY")
            && every > 0
            && frame.is_multiple_of(every)
        {
            invalidations.write(InvalidateShadowCache::All);
        }
        if frame == 1 || scenario(frame) != scenario(frame.saturating_sub(1)) {
            info!("oracle: frame {frame}: {}", scenario(frame));
        }
        let mut cube = |size: f32| Mesh3d(meshes.add(Cuboid::from_length(size)));
        let names = script.named.clone();
        let named = |name: &str| names[name];
        let script_lights = script.lights.clone();
        let light_position = |lights: &ScriptLights, index: usize| -> Vec3 {
            lights
                .get(script_lights[index])
                .map(|(t, ..)| t.translation)
                .unwrap()
        };
        let nudge = |casters: &mut ScriptCasters, name: &str, by: Vec3| {
            if let Ok(mut transform) = casters.get_mut(named(name)) {
                transform.translation += by;
            }
        };
        let step = Vec3::X * 0.05;
        match frame {
            80 => {
                for entity in take(&mut script, "warm") {
                    commands.entity(entity).despawn();
                }
            }
            100 => {
                let fox = spawn_gltf(
                    &mut commands,
                    &asset_server,
                    &mut graphs,
                    "models/animated/Fox.glb",
                    Transform::from_xyz(1.0, 0.0, -0.5).with_scale(Vec3::splat(0.012)),
                );
                commands.entity(fox).insert(Circling {
                    center: Vec3::new(0.5, 0.0, -0.5),
                    radius: 1.0,
                    speed: 1.2,
                });
                script.spawned.push(("fox", fox));
            }
            130 => {
                let morph = spawn_gltf(
                    &mut commands,
                    &asset_server,
                    &mut graphs,
                    "models/animated/MorphStressTest.gltf",
                    Transform::from_xyz(-0.5, 0.4, 0.5).with_scale(Vec3::splat(0.4)),
                );
                script.spawned.push(("morph", morph));
            }
            160 => {
                let material = materials.add(Color::srgb(0.9, 0.3, 0.3));
                for i in 0..20 {
                    let t = i as f32;
                    let burst = commands
                        .spawn((
                            cube(0.2),
                            MeshMaterial3d(material.clone()),
                            Transform::from_xyz(
                                -1.0 + (t * 0.37) % 3.0,
                                0.2 + (t * 0.13) % 1.0,
                                -2.0 + (t * 0.71) % 3.0,
                            ),
                            Velocity(Vec3::new(0.6, 0.0, 0.4)),
                        ))
                        .id();
                    script.spawned.push(("burst", burst));
                }
            }
            190 => {
                for entity in take(&mut script, "burst") {
                    commands.entity(entity).despawn();
                }
            }
            210 | 240 => {
                let light = script.lights[5];
                if let Ok((_, mut visibility, _)) = lights.get_mut(light) {
                    *visibility = if frame == 210 {
                        Visibility::Hidden
                    } else {
                        Visibility::Inherited
                    };
                }
            }
            260 => {
                if let Ok((mut transform, ..)) = lights.get_mut(script.lights[6]) {
                    transform.translation.x += 0.3;
                }
            }
            280 => {
                let position = light_position(&lights, 9);
                let near = commands
                    .spawn((
                        cube(0.3),
                        MeshMaterial3d(materials.add(Color::WHITE)),
                        Transform::from_translation(position - Vec3::Y * 0.4),
                        Velocity(Vec3::Y * 1.2),
                    ))
                    .id();
                script.spawned.push(("near", near));
            }
            300 => {
                for entity in take(&mut script, "near") {
                    commands.entity(entity).despawn();
                }
            }
            320 => {
                for entity in take(&mut script, "fox")
                    .into_iter()
                    .chain(take(&mut script, "morph"))
                    .chain(&orbiting)
                {
                    commands.entity(entity).despawn();
                }
            }
            340 => {
                let position = light_position(&lights, 10);
                let static_cube = commands
                    .spawn((
                        cube(0.4),
                        MeshMaterial3d(materials.add(Color::WHITE)),
                        Transform::from_xyz(position.x, 0.4, position.z),
                    ))
                    .id();
                script.spawned.push(("static", static_cube));
            }
            360 => {
                for entity in take(&mut script, "static") {
                    commands.entity(entity).despawn();
                }
            }
            380 => {
                // A light added and another removed: the shadow map layers
                // of the lights after them move.
                let spot_light = *lights.get(script.lights[0]).unwrap().2;
                commands.spawn((
                    spot_light,
                    Transform::from_xyz(2.5, 2.0, 1.5)
                        .looking_at(Vec3::new(2.5, 0.0, 1.5), Vec3::X),
                ));
                let removed = script.lights.remove(12);
                commands.entity(removed).despawn();
            }
            LOADED_AT => loading.0 = false,
            420 | 452 | 453 => nudge(&mut casters, "door", step),
            440 => {
                // New meshes: their shadow pipelines wait (the readiness
                // delay), so the rebuilds of their lights wait too.
                let material = MeshMaterial3d(materials.add(Color::srgb(0.8, 0.8, 0.3)));
                let late = commands
                    .spawn((
                        cube(0.2),
                        material.clone(),
                        Transform::from_translation(under(
                            light_position(&lights, 1),
                            Vec3::new(0.0, 0.0, 0.35),
                        )),
                    ))
                    .id();
                let back = lights
                    .get(named("back_light"))
                    .map(|(t, ..)| t.translation)
                    .unwrap();
                let back_cube = commands
                    .spawn((
                        cube(0.3),
                        material,
                        Transform::from_translation(under(back, Vec3::ZERO)),
                    ))
                    .id();
                script.named.insert("late", late);
                script.named.insert("back_cube", back_cube);
            }
            455 => {
                // Circles over the door while it settles.
                let center = casters.get(named("door")).unwrap().translation;
                let circler = commands
                    .spawn((
                        cube(0.15),
                        MeshMaterial3d(materials.add(Color::srgb(0.9, 0.5, 0.1))),
                        Transform::from_translation(center + Vec3::Y * 0.3),
                        Circling {
                            center: center + Vec3::Y * 0.3,
                            radius: 0.15,
                            speed: 2.0,
                        },
                    ))
                    .id();
                script.spawned.push(("circler", circler));
            }
            479 => nudge(&mut casters, "settler", step),
            480 => {
                for entity in take(&mut script, "circler") {
                    commands.entity(entity).despawn();
                }
            }
            490 => {
                // The light, a promotion and the settler's settle, all in
                // this frame.
                if let Ok((mut transform, ..)) = lights.get_mut(script.lights[2]) {
                    transform.rotate_local_x(0.02);
                }
                nudge(&mut casters, "parent", step);
            }
            520 => {
                shadow_map.size = 1024;
                nudge(&mut casters, "door", step);
            }
            540 => {
                commands
                    .entity(script.lights[3])
                    .insert(RenderLayers::from_layers(&[0, 1]));
                // A caster only that light's layer 1 sees.
                let layer_one = commands
                    .spawn((
                        cube(0.3),
                        MeshMaterial3d(materials.add(Color::WHITE)),
                        Transform::from_translation(under(
                            light_position(&lights, 3),
                            Vec3::new(0.0, 0.0, -0.3),
                        )),
                        RenderLayers::layer(1),
                    ))
                    .id();
                script.spawned.push(("layer_one", layer_one));
            }
            560 => shadow_map.size = 2048,
            570 => {
                commands
                    .entity(named("child"))
                    .insert(ChildOf(named("parent2")));
            }
            579 => script.promotions_before = counters.promotions,
            580..600 => {
                // The same values written again: nothing moves.
                for mut transform in &mut movables {
                    transform.set_changed();
                }
                for name in ["door", "jitter_a", "micro"] {
                    if let Ok(mut transform) = casters.get_mut(named(name)) {
                        transform.set_changed();
                    }
                }
            }
            _ => {}
        }
        if frame == 600
            && let Ok(mut transform) = casters.get_mut(named("micro"))
        {
            transform.translation.x = transform.translation.x.next_up();
        }
        if (600..640).contains(&frame) {
            nudge(&mut casters, "drifter", Vec3::X * 1e-4);
        }
        if (600..690).contains(&frame) && (frame - 600).is_multiple_of(SETTLE - 2) {
            nudge(&mut casters, "jitter_a", Vec3::X * 1e-3);
        }
        if (600..690).contains(&frame) && (frame - 600).is_multiple_of(15) {
            nudge(&mut casters, "jitter_b", Vec3::X * 1e-3);
        }
        match frame {
            600 => {
                // The identical writes promoted nothing.
                let promotions = counters.promotions - script.promotions_before;
                script.checks += 1;
                if promotions != 0 {
                    script.failed_checks.push((
                        frame,
                        format!("{promotions} promotions from identical writes"),
                    ));
                }
            }
            690 => {
                commands
                    .entity(named("forced"))
                    .insert(VisibilityRange::abrupt(0.0, 1000.0));
            }
            700 => {
                commands.entity(named("forced")).remove::<VisibilityRange>();
            }
            720 => {
                commands.entity(named("forced")).insert(ShadowCasterLive);
            }
            730 => {
                commands
                    .entity(named("forced"))
                    .remove::<ShadowCasterLive>();
            }
            750 => {
                // Debris thrown under four lights, stopping together.
                let material = MeshMaterial3d(materials.add(Color::srgb(0.6, 0.3, 0.2)));
                for i in 0..30 {
                    let t = i as f32;
                    let light = light_position(&lights, 1 + i % 4);
                    let debris = commands
                        .spawn((
                            cube(0.1),
                            material.clone(),
                            Transform::from_translation(under(
                                light,
                                Vec3::new((t * 0.13) % 0.6 - 0.3, 0.2, (t * 0.29) % 0.6 - 0.3),
                            )),
                            Velocity(Vec3::new(0.2, -0.3, 0.1)),
                        ))
                        .id();
                    script.spawned.push(("debris", debris));
                }
            }
            760 => {
                for (name, entity) in &script.spawned {
                    if *name == "debris" {
                        commands.entity(*entity).remove::<Velocity>();
                    }
                }
            }
            822 => {
                // Still cubes that live shorter than the settle time: no
                // static layer may see them.
                script.counters_before = Some(*counters);
                let material = MeshMaterial3d(materials.add(Color::srgb(0.9, 0.9, 0.2)));
                for i in 0..5 {
                    let light = light_position(&lights, 1 + i % 4);
                    let short = commands
                        .spawn((
                            cube(0.2),
                            material.clone(),
                            Transform::from_translation(under(light, Vec3::new(0.3, 0.0, -0.3))),
                        ))
                        .id();
                    script.spawned.push(("short", short));
                }
            }
            828 => {
                for entity in take(&mut script, "short") {
                    commands.entity(entity).despawn();
                }
            }
            840 => {
                script.checks += 1;
                if let Some(before) = script.counters_before {
                    // (`SPOTLIGHT_ORACLE_INVALIDATE_EVERY` asks for rebuilds of
                    // its own.)
                    let rebuilds = if env::<u32>("SPOTLIGHT_ORACLE_INVALIDATE_EVERY").is_some() {
                        0
                    } else {
                        counters.rebuild_requests - before.rebuild_requests
                    };
                    let settles = (counters.first_settles + counters.demotions)
                        - (before.first_settles + before.demotions);
                    if rebuilds != 0 || settles != 0 {
                        script.failed_checks.push((
                            frame,
                            format!("short-lived spawns: {rebuilds} rebuilds, {settles} settles"),
                        ));
                    }
                }
            }
            845 => {
                // Live casters under four lights, moving every frame.
                let material = MeshMaterial3d(materials.add(Color::srgb(0.3, 0.6, 0.9)));
                for index in 1..5 {
                    let center = under(light_position(&lights, index), Vec3::Y * 0.2);
                    let circler = commands
                        .spawn((
                            cube(0.15),
                            material.clone(),
                            Transform::from_translation(center),
                            Circling {
                                center,
                                radius: 0.3,
                                speed: 1.5 + index as f32 * 0.2,
                            },
                        ))
                        .id();
                    script.spawned.push(("away_circler", circler));
                }
            }
            850 | 881 | 900 => {
                // Every light out of view: the camera far off, looking away.
                if let Ok(mut transform) = camera.single_mut() {
                    *transform = Transform::from_xyz(0.0, 5.0, 60.0).looking_to(Vec3::Z, Vec3::Y);
                }
            }
            858 | 883 | 912 => {
                if let Ok(mut transform) = camera.single_mut() {
                    *transform =
                        Transform::from_xyz(-4.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
                }
                if frame == 858 {
                    // A settled caster moves in the frame its light is back.
                    nudge(&mut casters, "door", step);
                }
            }
            870 => {
                // Settles after the settle time; its light's rebuild then
                // waits for its pipeline (the readiness delay) while the light
                // goes out of view and comes back.
                let late2 = commands
                    .spawn((
                        cube(0.2),
                        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.4, 0.8))),
                        Transform::from_translation(under(
                            light_position(&lights, 2),
                            Vec3::new(0.35, 0.0, 0.0),
                        )),
                    ))
                    .id();
                script.named.insert("late2", late2);
            }
            904 => {
                for entity in take(&mut script, "away_circler") {
                    commands.entity(entity).despawn();
                }
            }
            915 => {
                // Removed and inserted again under the same id (an `Added`
                // event, no `Modified`): the same vertices and bounds, without
                // the top and bottom faces.
                let id = script.swap_mesh.id();
                if let Some(mut mesh) = meshes.remove(id) {
                    if let Some(Indices::U32(indices)) = mesh.indices_mut() {
                        indices.truncate(24);
                    }
                    meshes.insert(id, mesh).unwrap();
                }
            }
            930 => {
                // The same for a material: now it casts no shadow.
                let id = script.swap_material.id();
                if materials.remove(id).is_some() {
                    materials
                        .insert(
                            id,
                            StandardMaterial {
                                base_color: Color::srgba(0.7, 0.7, 0.9, 0.0),
                                alpha_mode: AlphaMode::Mask(0.5),
                                ..default()
                            },
                        )
                        .unwrap();
                }
            }
            800 => {
                let material = MeshMaterial3d(materials.add(Color::srgb(0.5, 0.5, 0.7)));
                let light = light_position(&lights, 7);
                for i in 0..5 {
                    let built = commands
                        .spawn((
                            cube(0.15),
                            material.clone(),
                            Transform::from_translation(under(
                                light,
                                Vec3::new(i as f32 * 0.16 - 0.32, -0.1, 0.3),
                            )),
                        ))
                        .id();
                    if i == 0 {
                        script.named.insert("built", built);
                    }
                }
            }
            _ => {}
        }
    }

    fn move_movers(
        time: Res<Time>,
        mut movers: Query<(&mut Transform, Option<&Velocity>, Option<&Circling>)>,
    ) {
        for (mut transform, velocity, circling) in &mut movers {
            if let Some(velocity) = velocity {
                transform.translation += velocity.0 * time.delta_secs();
            }
            if let Some(circling) = circling {
                let angle = time.elapsed_secs() * circling.speed;
                transform.translation = circling.center
                    + Vec3::new(ops::cos(angle), 0.0, ops::sin(angle)) * circling.radius;
                transform.rotation = Quat::from_rotation_y(-angle);
            }
        }
    }

    /// After the frame's classification: the named casters are settled or
    /// live as [`expectations`] says, and the debris settled together.
    fn check_expectations(
        frame: Res<FrameCount>,
        mut script: ResMut<Script>,
        settled: Query<(), With<StaticShadowCaster>>,
        counters: Res<ShadowCacheCounters>,
        feedback: Res<ShadowCacheFeedback>,
        mut expectations_list: Local<Option<Vec<Expect>>>,
    ) {
        if !cache_on() || hashes_file().is_some() {
            return;
        }
        let frame = frame.0;
        let list = expectations_list.get_or_insert_with(expectations);
        let mut failed = Vec::new();
        let mut checks = 0;
        for expect in list.iter().filter(|expect| expect.frame == frame) {
            let Some(entity) = script.named.get(expect.name) else {
                continue;
            };
            checks += 1;
            if settled.contains(*entity) != expect.settled {
                failed.push((
                    frame,
                    format!(
                        "{} should be {} ({:?})",
                        expect.name,
                        if expect.settled { "settled" } else { "live" },
                        *counters,
                    ),
                ));
            }
        }
        // Out of view, the lights with movers near (here every light: each
        // one's range holds the whole scene) draw nothing; back in view, none
        // is dormant.
        if matches!(frame, 856 | 868) {
            checks += 1;
            let dormant = feedback
                .snapshot()
                .values()
                .filter(|status| status.state == ShadowCacheState::Dormant)
                .count();
            if (frame == 856 && dormant < 4) || (frame == 868 && dormant > 0) {
                failed.push((frame, format!("{dormant} lights dormant")));
            }
        }
        if frame == 790 {
            checks += 1;
            let live = script
                .spawned
                .iter()
                .filter(|(name, entity)| *name == "debris" && !settled.contains(*entity))
                .count();
            if live > 0 {
                failed.push((frame, format!("{live} debris still live")));
            }
        }
        script.checks += checks;
        script.failed_checks.extend(failed);
    }

    /// Prints what the oracle found, and ends the run.
    ///
    /// It passes when every frame from [`check_from`] to [`check_to`] has a
    /// record for each light-frame the audit expected, each one compared
    /// without a differing texel or skipped as dormant (a light that lights
    /// no view), none missing, and every state check held.
    fn collect_results(
        frame: Res<FrameCount>,
        results: Res<ShadowCacheAuditResults>,
        mut script: ResMut<Script>,
        counters: Res<ShadowCacheCounters>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let checked = check_from()..=check_to();
        for record in results.take() {
            if !checked.contains(&record.frame) {
                continue;
            }
            if hashes_file().is_some() {
                let plan = format!(
                    "{:?} movers {} written {} restored {}",
                    record.plan,
                    record.movers_texels,
                    record.written_texels,
                    record.restored_texels
                );
                // A dormant light-frame is listed without a hash (the
                // comparison skips it), a missing one as `missing`.
                let hash = match (record.status, record.hash) {
                    (ShadowCacheAuditStatus::Compared, Some(hash)) => format!("{hash:016x}"),
                    (ShadowCacheAuditStatus::Dormant, _) => "dormant".to_string(),
                    _ => "missing".to_string(),
                };
                script
                    .hashes
                    .push((record.frame, record.light, hash, record.cached, plan));
                continue;
            }
            match record.status {
                ShadowCacheAuditStatus::Missing(why) => warn!(
                    "oracle: frame {} ({}): light {:?} ({:?}) not compared: {why}",
                    record.frame,
                    scenario(record.frame),
                    record.light,
                    record.plan,
                ),
                _ if record.differing > 0 => warn!(
                    "oracle: frame {} ({}): light {:?} (layers {:?}, {:?}): {} texels differ, first at {:?}",
                    record.frame,
                    scenario(record.frame),
                    record.light,
                    record.layers,
                    record.plan,
                    record.differing,
                    record.first,
                ),
                _ => {}
            }
            script.records.push(AuditedLightFrame {
                frame: record.frame,
                light: record.light,
                status: record.status,
                differing: record.differing,
                expected: record.expected,
            });
        }
        if frame.0 < exit_at() {
            return;
        }
        if let Some(path) = hashes_file() {
            let mut hashes = core::mem::take(&mut script.hashes);
            hashes.sort_unstable();
            // Never over an earlier run's file.
            let mut file = std::fs::File::create_new(&path)
                .unwrap_or_else(|error| panic!("the hashes file {path}: {error}"));
            for (frame, light, hash, cached, plan) in &hashes {
                writeln!(file, "{frame} {light} {hash} {} {plan}", u8::from(*cached)).unwrap();
            }
            let frames = {
                let mut frames: Vec<_> = hashes.iter().map(|(frame, ..)| *frame).collect();
                frames.dedup();
                frames.len()
            };
            info!(
                "oracle: HASHES: {} light-frames in {frames} frames written to {path}; {:?}",
                hashes.len(),
                *counters
            );
            exit.write(AppExit::Success);
            return;
        }
        let records = &script.records;
        let mut per_frame: HashMap<u32, (u32, u32)> = HashMap::default();
        for record in records {
            let entry = per_frame
                .entry(record.frame)
                .or_insert((0, record.expected));
            entry.0 += 1;
        }
        let incomplete: Vec<u32> = {
            let mut frames: Vec<_> = checked
                .clone()
                .filter(|frame| {
                    per_frame
                        .get(frame)
                        .is_none_or(|&(found, expected)| expected == 0 || found != expected)
                })
                .collect();
            frames.sort_unstable();
            frames
        };
        let compared = records
            .iter()
            .filter(|record| record.status == ShadowCacheAuditStatus::Compared)
            .count();
        let dormant = records
            .iter()
            .filter(|record| record.status == ShadowCacheAuditStatus::Dormant)
            .count();
        let missing = records.len() - compared - dormant;
        let mut failing_frames: Vec<_> = records
            .iter()
            .filter(|record| {
                matches!(record.status, ShadowCacheAuditStatus::Missing(_))
                    || (record.status == ShadowCacheAuditStatus::Compared && record.differing > 0)
            })
            .map(|record| record.frame)
            .collect();
        let failing = failing_frames.len();
        failing_frames.sort_unstable();
        failing_frames.dedup();
        for (frame, what) in &script.failed_checks {
            warn!("oracle: frame {frame} ({}): {what}", scenario(*frame));
        }
        if let Some(first) = incomplete.first() {
            warn!(
                "oracle: {} frames without a record for every expected light, first {first} ({})",
                incomplete.len(),
                scenario(*first)
            );
        }
        let lights: usize = {
            let mut lights: Vec<_> = records.iter().map(|record| record.light).collect();
            lights.sort_unstable();
            lights.dedup();
            lights.len()
        };
        let pass = failing == 0 && incomplete.is_empty() && script.failed_checks.is_empty();
        info!(
            "oracle: {}: {} frames complete of {}, {} light-frames of {lights} lights ({compared} compared, {dormant} dormant skipped, {missing} missing), {} failing light-frames in {} frames{}; {} of {} state checks failed; {:?}",
            if pass { "PASS" } else { "FAIL" },
            per_frame.len() - incomplete.iter().filter(|frame| per_frame.contains_key(*frame)).count(),
            check_to() - check_from() + 1,
            records.len(),
            failing,
            failing_frames.len(),
            failing_frames
                .first()
                .map(|frame| format!(", first at frame {frame} ({})", scenario(*frame)))
                .unwrap_or_default(),
            script.failed_checks.len(),
            script.checks,
            *counters,
        );
        exit.write(if pass {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}

/// A scene generator for the spot light shadow cache at scale (`--stress`).
///
/// A level of 200 x 200 m with spot lights on a grid, moving cubes wandering
/// over the whole level and cubes that stand still, in a small window on top
/// with vsync. It runs a number of frames, logs what the cached lights did,
/// and exits. With `--oracle` it runs the oracle's checks on this scene
/// instead of the scripted one.
///
/// Environment (`SPOTLIGHT_STRESS_*`): `LIGHTS` (4), `MOVERS` (0), `STATICS`
/// (0), `MAP` (shadow map size, 2048), `NEAR=1` (a mover across each light's
/// near plane: every tile restored every frame), `RECT=size` (a still cube
/// of this size under each light: its tiles restored every frame), `FRAMES`
/// (1200), `TIMING=n` (logs the median and 95th percentile of the main
/// schedule, the extraction and the render schedule over the last n frames,
/// and the cache's counters).
mod stress {
    use super::*;
    use bevy::{
        app::{TaskPoolOptions, TaskPoolPlugin},
        diagnostic::FrameCount,
        light::{DirectionalLightShadowMap, ShadowCacheCounters},
        platform::time::Instant,
        render::{Render, RenderApp, RenderSystems},
        window::{PresentMode, WindowLevel, WindowResolution},
    };
    use std::{
        str::FromStr,
        sync::{Arc, Mutex},
    };

    pub fn env<T: FromStr>(name: &str, default: T) -> T {
        std::env::var(format!("SPOTLIGHT_STRESS_{name}"))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    const HALF_LEVEL: f32 = 100.0;

    /// The restored texels and the plans of the cached lights, per frame.
    #[derive(Resource, Default)]
    struct Run {
        restored_texels: Vec<u64>,
        states: std::collections::BTreeMap<String, u64>,
    }

    /// CPU times per frame (ms): main schedule, extraction, render schedule.
    #[derive(Resource, Clone, Default)]
    struct Times(Arc<Mutex<TimesState>>);

    #[derive(Default)]
    struct TimesState {
        main_begin: Option<Instant>,
        render_begin: Option<Instant>,
        main: Vec<f64>,
        extract: Vec<f64>,
        render: Vec<f64>,
    }

    fn since(begin: Option<Instant>) -> f64 {
        begin.map_or(0.0, |begin| begin.elapsed().as_secs_f64() * 1e3)
    }

    /// Times the main schedule, the extraction and the render schedule.
    fn add_timing(app: &mut App) {
        let times = Times::default();
        app.insert_resource(times.clone())
            .add_systems(First, |t: Res<Times>| {
                t.0.lock().unwrap().main_begin = Some(Instant::now());
            })
            .add_systems(Last, |t: Res<Times>| {
                let mut state = t.0.lock().unwrap();
                let main = since(state.main_begin);
                state.main.push(main);
            });
        let render = app.sub_app_mut(RenderApp);
        if let Some(mut extract) = render.take_extract() {
            let extract_times = times.clone();
            render.set_extract(move |main_world, render_world| {
                let begin = Instant::now();
                extract(main_world, render_world);
                extract_times
                    .0
                    .lock()
                    .unwrap()
                    .extract
                    .push(since(Some(begin)));
            });
        }
        render.insert_resource(times).add_systems(
            Render,
            (
                (|t: Res<Times>| t.0.lock().unwrap().render_begin = Some(Instant::now()))
                    .in_set(RenderSystems::ExtractCommands),
                (|t: Res<Times>| {
                    let mut state = t.0.lock().unwrap();
                    let render = since(state.render_begin);
                    state.render.push(render);
                })
                .in_set(RenderSystems::PostCleanup),
            ),
        );
    }

    /// The median and the 95th percentile of the last `n` values.
    fn percentiles(values: &[f64], n: usize) -> (f64, f64) {
        let mut last: Vec<f64> = values[values.len().saturating_sub(n)..].to_vec();
        if last.is_empty() {
            return (0.0, 0.0);
        }
        last.sort_by(f64::total_cmp);
        let at = |q: f64| last[((last.len() - 1) as f64 * q).round() as usize];
        (at(0.5), at(0.95))
    }

    #[derive(Component)]
    pub struct Wander {
        center: Vec3,
        radius: f32,
        speed: f32,
        phase: f32,
    }

    pub fn run(oracle: bool) -> AppExit {
        let mut task_pool_options = TaskPoolOptions::default();
        // As MPS's stands (`mps_devtools::gpu::COMPUTE_THREADS`).
        task_pool_options.compute.max_threads = 4;
        let map: usize = env("MAP", 2048);
        let lights: usize = env("LIGHTS", 4);
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "spotlight --stress".into(),
                        resolution: WindowResolution::new(640, 360),
                        present_mode: PresentMode::AutoVsync,
                        window_level: WindowLevel::AlwaysOnTop,
                        ..default()
                    }),
                    ..default()
                })
                .set(TaskPoolPlugin { task_pool_options }),
        )
        .insert_resource(DirectionalLightShadowMap { size: map })
        .insert_resource(ShadowCacheSettings {
            enabled: true,
            // Every light gets a static layer.
            budget_bytes: (lights * map * map * 4) as u64,
            ..default()
        })
        .init_resource::<Run>()
        .add_systems(Startup, spawn)
        .add_systems(Update, (wander, report).chain());
        if env("TIMING", 0usize) > 0 {
            add_timing(&mut app);
        }
        if oracle {
            app.add_plugins(oracle::OraclePlugin { scripted: false });
        }
        app.run()
    }

    pub fn spawn(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        let lights: usize = env("LIGHTS", 4);
        let movers: usize = env("MOVERS", 0);
        let statics: usize = env("STATICS", 0);
        let mut rng = ChaCha8Rng::seed_from_u64(19878367467713);
        let cube = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
        let gray = materials.add(Color::srgb(0.6, 0.6, 0.6));
        let blue = materials.add(Color::srgb_u8(124, 144, 255));
        let orange = materials.add(Color::srgb_u8(255, 200, 80));

        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(220.0, 220.0))),
            MeshMaterial3d(gray.clone()),
        ));

        let columns = (lights as f32).sqrt().ceil() as usize;
        let rows = lights.div_ceil(columns.max(1));
        let spacing = 2.0 * HALF_LEVEL / columns.max(rows) as f32;
        let mut light_positions = Vec::new();
        for index in 0..lights {
            let x = (index % columns) as f32 * spacing - HALF_LEVEL + spacing * 0.5;
            let z = (index / columns) as f32 * spacing - HALF_LEVEL + spacing * 0.5;
            let position = Vec3::new(x, 8.0, z);
            light_positions.push(position);
            commands.spawn((
                SpotLight {
                    intensity: 400_000.0,
                    range: 20.0,
                    shadow_maps_enabled: true,
                    inner_angle: PI / 4.0 * 0.85,
                    outer_angle: PI / 4.0,
                    ..default()
                },
                Transform::from_translation(position).looking_at(Vec3::new(x, 0.0, z), Vec3::X),
            ));
        }

        let near: u32 = env("NEAR", 0);
        let rect: f32 = env("RECT", 0.0);
        for position in &light_positions {
            if near != 0 {
                // From 0.07 to 0.17 m in front of the light, inside its cone;
                // it moves, so it stays live.
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::from_length(0.1))),
                    MeshMaterial3d(orange.clone()),
                    Transform::from_translation(*position + Vec3::new(0.06, -0.12, 0.0)),
                    Wander {
                        center: *position + Vec3::new(0.06, -0.12, 0.0),
                        radius: 0.001,
                        speed: 1.0,
                        phase: 0.0,
                    },
                ));
            }
            if rect > 0.0 {
                // A still cube, kept live: its tiles are restored every
                // frame.
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::from_length(rect))),
                    MeshMaterial3d(orange.clone()),
                    Transform::from_xyz(position.x, rect * 0.5, position.z),
                    bevy::light::ShadowCasterLive,
                ));
            }
        }

        let mut random_ground = || {
            Vec3::new(
                rng.random_range(-HALF_LEVEL..HALF_LEVEL),
                0.25,
                rng.random_range(-HALF_LEVEL..HALF_LEVEL),
            )
        };
        let static_positions: Vec<Vec3> = (0..statics).map(|_| random_ground()).collect();
        let mover_starts: Vec<Vec3> = (0..movers).map(|_| random_ground()).collect();
        commands.spawn_batch(static_positions.into_iter().map({
            let cube = cube.clone();
            move |position| {
                (
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(blue.clone()),
                    Transform::from_translation(position),
                )
            }
        }));
        commands.spawn_batch(
            mover_starts
                .into_iter()
                .enumerate()
                .map(move |(index, center)| {
                    let t = index as f32;
                    (
                        Mesh3d(cube.clone()),
                        MeshMaterial3d(orange.clone()),
                        Transform::from_translation(center),
                        Wander {
                            center,
                            radius: 2.0 + (t * 0.37) % 6.0,
                            speed: 0.3 + (t * 0.13) % 0.7,
                            phase: t,
                        },
                    )
                }),
        );

        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 140.0, 140.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }

    fn wander(time: Res<Time>, mut movers: Query<(&mut Transform, &Wander)>) {
        movers.par_iter_mut().for_each(|(mut transform, wander)| {
            let angle = wander.phase + time.elapsed_secs() * wander.speed;
            transform.translation =
                wander.center + Vec3::new(ops::cos(angle), 0.0, ops::sin(angle)) * wander.radius;
        });
    }

    fn report(
        mut run: ResMut<Run>,
        frame: Res<FrameCount>,
        feedback: Res<ShadowCacheFeedback>,
        counters: Res<ShadowCacheCounters>,
        times: Option<Res<Times>>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let statuses = feedback.snapshot();
        run.restored_texels
            .push(statuses.values().map(|status| status.restored_texels).sum());
        for status in statuses.values() {
            *run.states.entry(format!("{:?}", status.state)).or_insert(0) += 1;
        }
        if frame.0 < env("FRAMES", 1200) {
            return;
        }
        let frames = run.restored_texels.len().max(1) as f64;
        info!(
            "stress: lights {} movers {} statics {} map {}: {:.0} texels restored per frame; light-frames {:?}; {:?}",
            env("LIGHTS", 4usize),
            env("MOVERS", 0usize),
            env("STATICS", 0usize),
            env("MAP", 2048usize),
            run.restored_texels.iter().sum::<u64>() as f64 / frames,
            run.states,
            *counters,
        );
        if let Some(times) = times {
            let n = env("TIMING", 0usize);
            let state = times.0.lock().unwrap();
            let (main, main95) = percentiles(&state.main, n);
            let (extract, extract95) = percentiles(&state.extract, n);
            let (render, render95) = percentiles(&state.render, n);
            info!(
                "stress timing over {n} frames: main {main:.4} (p95 {main95:.4}) extract {extract:.4} (p95 {extract95:.4}) render {render:.4} (p95 {render95:.4}) ms"
            );
        }
        exit.write(AppExit::Success);
    }
}
