//! A sun whose far shadow cascade is cached: a stable far projection
//! ([`StableFarCascade`]) that only a camera translation beyond a guard
//! moves, the shadow cache on top of it, and a terrain whose camera patches
//! swap level of detail as the camera moves while the far cascade draws
//! stable stand-ins ([`ShadowRepresentation`]).
//!
//! `--oracle` runs a script instead of the free camera and checks every
//! cached far cascade against a full redraw of its exact projection, texel
//! by texel (`ShadowCacheAuditPlugin`'s reference views), plus state checks:
//! the far projection survives turns and moves only beyond its guard, its
//! coverage holds the camera's far slice in every orientation, a moving sun
//! is never cached, camera LOD swaps never redraw the far cascade, stand-ins
//! never reach the camera or the near cascades. `SHADOW_CACHE_FAULT=NAME`
//! injects a fault the oracle must catch; `SUN_ORACLE_CACHE=0` runs with the
//! cache off, `SUN_ORACLE_STAND_INS=0` without stand-ins.
//!
//! For look reviews: `SUN_STABLE=0` uses Bevy's fitted far cascade instead of
//! the stable one, `SUN_CAPTURE=DIR` saves the main view at fixed frames of
//! the oracle's script (stills, and a strip of consecutive walk frames), and
//! `SUN_RESOLUTION=WxH` sets the window's size.

use std::f32::consts::*;

use bevy::{
    asset::RenderAssetUsages,
    camera::{visibility::VisibleEntities, Viewport},
    diagnostic::FrameCount,
    light::{
        CascadeShadowConfigBuilder, Cascades, DirectionalLightShadowCache,
        DirectionalLightShadowMap, ShadowCacheCounters, ShadowCacheFaults, ShadowCacheFeedback,
        ShadowCacheLoading, ShadowCacheSettings, ShadowRepresentation, ShadowSubview,
        StableFarCascade, StableFarProjections,
    },
    math::ops,
    mesh::{Indices, PrimitiveTopology},
    pbr::{
        ShadowCacheAuditPlugin, ShadowCacheAuditResults, ShadowCacheAuditSettings,
        ShadowCacheAuditStatus,
    },
    platform::collections::HashMap,
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    time::TimeUpdateStrategy,
    window::{PresentMode, WindowResolution},
};

/// One frame of simulation time.
const FRAME: std::time::Duration = std::time::Duration::from_nanos(33_333_333);
/// The settle time, in frames.
const SETTLE: u32 = 10;
/// The scene counts as loaded from this frame.
const LOADED_AT: u32 = 40;
/// The first and last checked frames, and the last frame.
const CHECK_FROM: u32 = 70;
const CHECK_TO: u32 = 980;
const EXIT_AT: u32 = 1000;

/// Terrain tiles: TILES x TILES of TILE meters, centered on the origin.
const TILES: i32 = 8;
const TILE: f32 = 60.0;

/// The script's segments (first frame of each).
const TURN: u32 = 120;
const WALK: u32 = 220;
const CUT: u32 = 340;
const SUN_STEP: u32 = 420;
const SUN_MOVE: u32 = 480;
const SUN_STOP: u32 = 560;
const CASTER_MOVE: u32 = 620;
const CASTER_SPAWN: u32 = 660;
const STAND_INS_OFF: u32 = 720;
const CASTER_DESPAWN: u32 = 780;
const FLY: u32 = 820;
/// The camera holds and every mover stops: they settle, and the far cascade
/// has nothing live (Kept), until QUIET_END, when they move on (promotions).
const QUIET: u32 = 900;
const QUIET_END: u32 = 950;
/// The second camera is off from here to STAND_INS_OFF: the main camera owns the
/// far layer, so its cached cascade restores tiles (Movers) instead of the
/// whole layer.
const ONE_CAMERA: u32 = 590;
/// In the single-camera stretch: a fast mover jumps 40 m, then the other goes.
const JUMP: u32 = 640;
const GONE: u32 = 680;

fn main() -> AppExit {
    let oracle = std::env::args().any(|arg| arg == "--oracle");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "shadow cache: sun".into(),
            resolution: resolution(),
            present_mode: PresentMode::AutoVsync,
            ..default()
        }),
        ..default()
    }))
    .insert_resource(DirectionalLightShadowMap { size: 1024 })
    .insert_resource(ShadowCacheSettings {
        enabled: env("SUN_ORACLE_CACHE").unwrap_or(1) != 0,
        settle_time: FRAME * SETTLE,
        ..default()
    })
    .init_resource::<Terrain>()
    .add_systems(Startup, setup)
    .add_systems(Update, (swap_terrain_lods, move_movers));
    if oracle {
        oracle::add(&mut app);
    } else {
        app.add_systems(Update, free_camera);
    }
    app.run()
}

/// The window's size: `SUN_RESOLUTION=WxH`, or 640x360.
fn resolution() -> WindowResolution {
    let size = std::env::var("SUN_RESOLUTION").ok().and_then(|value| {
        let (w, h) = value.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });
    let (w, h) = size.unwrap_or((640, 360));
    WindowResolution::new(w, h)
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// The main camera (the oracle's first).
#[derive(Component)]
struct MainCamera;

/// The second camera.
#[derive(Component)]
struct SecondCamera;

/// A fast mover the script teleports (0) or removes (1).
#[derive(Component)]
struct FastMover(u32);

/// A caster that moves every frame.
#[derive(Component)]
struct Mover {
    center: Vec3,
    radius: f32,
    speed: f32,
}

/// A static caster the script moves or removes.
#[derive(Component)]
struct Scripted(u32);

/// A camera patch of the terrain: it swaps between its fine and coarse mesh
/// by its distance to the main camera (the churn of a camera-selected LOD).
#[derive(Component)]
struct DetailTile {
    center: Vec3,
    fine: bool,
}

#[derive(Resource, Default)]
struct Terrain {
    fine: Vec<Handle<Mesh>>,
    coarse: Vec<Handle<Mesh>>,
}

fn height(x: f32, z: f32) -> f32 {
    3.0 * ops::sin(x / 13.0) * ops::cos(z / 17.0) + 2.0 * ops::sin((x + z) / 23.0)
}

/// A heightfield tile of `cells` x `cells` quads with its corner at
/// (`x0`, `z0`).
fn tile_mesh(x0: f32, z0: f32, cells: u32) -> Mesh {
    let step = TILE / cells as f32;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for j in 0..=cells {
        for i in 0..=cells {
            let (x, z) = (x0 + i as f32 * step, z0 + j as f32 * step);
            positions.push([x, height(x, z), z]);
            let e = 0.1;
            let n = Vec3::new(
                height(x - e, z) - height(x + e, z),
                2.0 * e,
                height(x, z - e) - height(x, z + e),
            )
            .normalize();
            normals.push(n.to_array());
        }
    }
    let mut indices = Vec::new();
    let row = cells + 1;
    for j in 0..cells {
        for i in 0..cells {
            let a = j * row + i;
            indices.extend([a, a + row, a + 1, a + 1, a + row, a + row + 1]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn stand_ins() -> bool {
    env("SUN_ORACLE_STAND_INS").unwrap_or(1) != 0
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut terrain: ResMut<Terrain>,
) {
    let ground = materials.add(Color::srgb(0.55, 0.6, 0.45));
    let stone = materials.add(Color::srgb(0.7, 0.7, 0.75));
    let moving = materials.add(Color::srgb(1.0, 0.7, 0.3));

    // The terrain: camera patches (detail, with two LODs) and far stand-ins.
    for tz in 0..TILES {
        for tx in 0..TILES {
            let x0 = (tx - TILES / 2) as f32 * TILE;
            let z0 = (tz - TILES / 2) as f32 * TILE;
            let fine = meshes.add(tile_mesh(x0, z0, 24));
            let coarse = meshes.add(tile_mesh(x0, z0, 6));
            terrain.fine.push(fine.clone());
            terrain.coarse.push(coarse.clone());
            let center = Vec3::new(x0 + TILE / 2.0, 0.0, z0 + TILE / 2.0);
            commands.spawn((
                Mesh3d(fine),
                MeshMaterial3d(ground.clone()),
                DetailTile { center, fine: true },
                ShadowRepresentation::Detail,
            ));
            commands.spawn((
                Mesh3d(meshes.add(tile_mesh(x0, z0, 4))),
                MeshMaterial3d(ground.clone()),
                ShadowRepresentation::FarCascadeStandIn,
            ));
        }
    }

    // Static casters: pillars and boxes over the terrain.
    let pillar = meshes.add(Cuboid::new(2.0, 14.0, 2.0));
    let block = meshes.add(Cuboid::new(6.0, 4.0, 3.0));
    for index in 0..48u32 {
        let angle = index as f32 * 2.399;
        let distance = 12.0 + 4.6 * index as f32;
        let (x, z) = (distance * ops::cos(angle), distance * ops::sin(angle));
        let (mesh, half_height) = if index % 3 == 0 {
            (block.clone(), 2.0)
        } else {
            (pillar.clone(), 7.0)
        };
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(stone.clone()),
            Transform::from_xyz(x, height(x, z) + half_height, z)
                .with_rotation(Quat::from_rotation_y(angle)),
            Scripted(index),
        ));
    }

    // Movers: live every frame.
    let mover = meshes.add(Cuboid::new(1.5, 1.5, 1.5));
    for index in 0..6 {
        let center = Vec3::new(-60.0 + index as f32 * 25.0, 0.0, 10.0 - index as f32 * 9.0);
        commands.spawn((
            Mesh3d(mover.clone()),
            MeshMaterial3d(moving.clone()),
            Transform::from_translation(center),
            Mover {
                center,
                radius: 4.0 + index as f32,
                speed: 0.6 + 0.15 * index as f32,
            },
        ));
    }
    // Two fast movers (about 15 m/s, just under a far-cascade texel a frame).
    // A tile mask's one-texel margin covers motion under a texel a frame, so
    // only a mover that jumps or goes leaves tiles its old shadow is in: the
    // script teleports one and removes the other (`JUMP`, `GONE`).
    for (index, (center, radius)) in [
        (Vec3::new(40.0, 0.0, -60.0), 30.0),
        (Vec3::new(-20.0, 0.0, 70.0), 24.0),
    ]
    .into_iter()
    .enumerate()
    {
        commands.spawn((
            FastMover(index as u32),
            Mesh3d(meshes.add(Cuboid::new(3.0, 3.0, 3.0))),
            MeshMaterial3d(moving.clone()),
            Transform::from_translation(center),
            Mover {
                center,
                radius,
                speed: 15.0 / radius,
            },
        ));
    }

    // The sun: three cascades to 200 m, the last one stable and cached.
    let sun = commands
        .spawn((
            DirectionalLight {
                illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, 0.7, -0.6, 0.0)),
            CascadeShadowConfigBuilder {
                num_cascades: 3,
                minimum_distance: 0.1,
                first_cascade_far_bound: 10.0,
                maximum_distance: 200.0,
                overlap_proportion: 0.2,
            }
            .build(),
        ))
        .id();
    if env("SUN_STABLE").unwrap_or(1) != 0 {
        commands.entity(sun).insert(StableFarCascade {
            stand_ins: stand_ins(),
            ..default()
        });
    }

    // The main camera and a second, smaller one elsewhere: both draw their
    // cascades into the same layers.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-30.0, 6.0, 40.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
        MainCamera,
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 1,
            viewport: Some(Viewport {
                physical_position: UVec2::new(440, 10),
                physical_size: UVec2::new(190, 110),
                ..default()
            }),
            ..default()
        },
        Transform::from_xyz(150.0, 40.0, -120.0).looking_at(Vec3::new(60.0, 0.0, -40.0), Vec3::Y),
        SecondCamera,
    ));
}

/// Swaps each camera patch between its LODs by its distance to the main
/// camera: a camera that moves swaps some of them almost every frame.
fn swap_terrain_lods(
    terrain: Res<Terrain>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    mut tiles: Query<(&mut DetailTile, &mut Mesh3d)>,
) {
    let Ok(camera) = camera.single() else {
        return;
    };
    let eye = camera.translation();
    for (index, (mut tile, mut mesh)) in tiles.iter_mut().enumerate() {
        let fine = Vec2::new(eye.x - tile.center.x, eye.z - tile.center.z).length() < 1.2 * TILE;
        if fine != tile.fine {
            tile.fine = fine;
            mesh.0 = if fine {
                terrain.fine[index].clone()
            } else {
                terrain.coarse[index].clone()
            };
        }
    }
}

fn move_movers(
    time: Res<Time>,
    frame: Res<FrameCount>,
    mut movers: Query<(&Mover, &mut Transform)>,
) {
    if (QUIET..QUIET_END).contains(&frame.0) {
        return;
    }
    let t = time.elapsed_secs();
    for (mover, mut transform) in &mut movers {
        let angle = t * mover.speed;
        let (x, z) = (
            mover.center.x + mover.radius * ops::cos(angle),
            mover.center.z + mover.radius * ops::sin(angle),
        );
        transform.translation = Vec3::new(x, height(x, z) + 1.0 + ops::sin(angle * 2.0), z);
        transform.rotation = Quat::from_rotation_y(angle);
    }
}

fn free_camera(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<&mut Transform, With<MainCamera>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };
    let speed = 20.0 * time.delta_secs();
    let forward = transform.forward().with_y(0.0).normalize_or_zero();
    let right = transform.right().with_y(0.0).normalize_or_zero();
    for (key, direction) in [
        (KeyCode::KeyW, forward),
        (KeyCode::KeyS, -forward),
        (KeyCode::KeyD, right),
        (KeyCode::KeyA, -right),
    ] {
        if keys.pressed(key) {
            transform.translation += direction * speed;
        }
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        transform.rotate_y(time.delta_secs());
    }
    if keys.pressed(KeyCode::ArrowRight) {
        transform.rotate_y(-time.delta_secs());
    }
}

mod oracle {
    use super::*;

    /// What the oracle collects.
    #[derive(Resource, Default)]
    struct Oracle {
        /// (frame, view, status, differing, expected, plan).
        records: Vec<(u32, ShadowViewId, ShadowCacheAuditStatus, u32, u32, String)>,
        /// Frames whose state checks failed, with what.
        failed: Vec<(u32, String)>,
        checks: u32,
        /// Per main-camera frame: the far projection's rebuild count.
        rebuilds: Vec<(u32, u64)>,
        /// The rebuild requests counted at the start of the walk.
        walk_requests: Option<u64>,
        scripted: Vec<(u32, Entity)>,
    }

    use bevy::light::ShadowViewId;

    pub fn add(app: &mut App) {
        let fault = std::env::var("SHADOW_CACHE_FAULT").unwrap_or_default();
        let faults = ShadowCacheFaults {
            skip_restore: fault == "skip_restore",
            drop_written: fault == "drop_written",
            drop_one_tile: fault == "drop_one_tile",
            miss_static_invalidation: fault == "miss_static_invalidation",
            miss_settle_invalidation: fault == "miss_settle_invalidation",
            stale_ack: fault == "stale_ack",
            wrong_layer: fault == "wrong_layer",
            acknowledge_early: fault == "acknowledge_early",
            miss_ownership: fault == "miss_ownership",
            stale_projection: fault == "stale_projection",
            wrong_cascade: fault == "wrong_cascade",
            miss_recenter: fault == "miss_recenter",
            ..default()
        };
        if !fault.is_empty() {
            assert_ne!(
                faults,
                ShadowCacheFaults::default(),
                "unknown fault {fault}"
            );
            info!("oracle: injected fault {fault}");
        }
        // Not faults: the fallback path, and late pipelines (longer than the
        // settle time, so a rebuild waits for them).
        let faults = ShadowCacheFaults {
            restore_not_ready: env("SUN_ORACLE_RESTORE_NOT_READY").unwrap_or(0) != 0,
            readiness_delay_frames: env("SUN_ORACLE_READINESS_DELAY").unwrap_or(15),
            ..faults
        };
        app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
            .insert_resource(faults)
            .insert_resource(ShadowCacheAuditSettings {
                enabled: true,
                fresh_redraws: true,
                hashes: false,
            })
            .add_plugins(ShadowCacheAuditPlugin)
            .init_resource::<Oracle>()
            .add_systems(Startup, list_scripted.after(setup))
            .add_systems(Update, script.before(swap_terrain_lods))
            .add_systems(
                PostUpdate,
                check_state.after(bevy::light::SimulationLightSystems::CheckLightVisibility),
            )
            .add_systems(Last, collect);
    }

    fn list_scripted(mut oracle: ResMut<Oracle>, casters: Query<(Entity, &Scripted)>) {
        oracle.scripted = casters.iter().map(|(entity, s)| (s.0, entity)).collect();
        oracle.scripted.sort_unstable();
    }

    /// The main camera's pose at a frame.
    fn camera_pose(frame: u32) -> Transform {
        let look = |eye: Vec3, yaw: f32, pitch: f32| {
            Transform::from_translation(eye).with_rotation(Quat::from_euler(
                EulerRot::YXZ,
                yaw,
                pitch,
                0.0,
            ))
        };
        let start = Vec3::new(-30.0, 6.0, 40.0);
        match frame {
            // Turns all the way round, looks up and down.
            f if (TURN..WALK).contains(&f) => {
                let t = (f - TURN) as f32 / (WALK - TURN) as f32;
                look(start, -0.6 + TAU * t, -0.15 + 0.6 * ops::sin(TAU * t))
            }
            // Walks 1.5 m a frame, past the guard: the far projection
            // re-centers, camera patches swap LODs.
            f if (WALK..CUT).contains(&f) => {
                let t = (f - WALK) as f32;
                look(start + Vec3::new(1.5 * t, 0.0, -0.4 * t), -0.6, -0.1)
            }
            // Cut to the other side, then hold.
            f if (CUT..FLY).contains(&f) => look(Vec3::new(120.0, 12.0, -100.0), 2.4, -0.2),
            // Flies fast along a line, then holds where it stopped.
            f if f >= FLY => {
                let t = (f.min(QUIET) - FLY) as f32;
                look(
                    Vec3::new(120.0 - 4.0 * t, 30.0, -100.0 + 3.0 * t),
                    2.0,
                    -0.3,
                )
            }
            _ => look(start, -0.6, -0.15),
        }
    }

    /// The sun's rotation at a frame: still, a step, a continuous move.
    fn sun(frame: u32) -> Quat {
        let elevation = match frame {
            f if f < SUN_STEP => 0.6,
            f if f < SUN_MOVE => 0.45,
            f if f < SUN_STOP => 0.45 + 0.004 * (f - SUN_MOVE) as f32,
            _ => 0.45 + 0.004 * (SUN_STOP - SUN_MOVE) as f32,
        };
        Quat::from_euler(EulerRot::YXZ, 0.7, -elevation, 0.0)
    }

    #[expect(clippy::too_many_arguments, reason = "a script")]
    fn script(
        frame: Res<FrameCount>,
        mut oracle: ResMut<Oracle>,
        mut commands: Commands,
        mut loading: ResMut<ShadowCacheLoading>,
        mut camera: Query<&mut Transform, With<MainCamera>>,
        mut second: Query<&mut Camera, With<SecondCamera>>,
        mut fast: Query<(Entity, &FastMover, &mut Mover)>,
        mut sun_light: Query<
            (&mut Transform, Option<&mut StableFarCascade>),
            (With<DirectionalLight>, Without<MainCamera>),
        >,
        mut casters: Query<
            &mut Transform,
            (
                With<Scripted>,
                Without<MainCamera>,
                Without<DirectionalLight>,
            ),
        >,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        counters: Res<ShadowCacheCounters>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let f = frame.0;
        if f == LOADED_AT {
            loading.0 = false;
        }
        if let Ok(mut transform) = camera.single_mut() {
            *transform = camera_pose(f);
        }
        if let Ok((mut transform, mut stable_far)) = sun_light.single_mut() {
            let rotation = sun(f);
            if transform.rotation != rotation {
                transform.rotation = rotation;
            }
            if f == STAND_INS_OFF
                && let Some(stable_far) = stable_far.as_mut()
                && stable_far.stand_ins
            {
                stable_far.stand_ins = false;
            }
        }
        if f == WALK {
            oracle.walk_requests = Some(counters.rebuild_requests);
        }
        for (entity, fast, mut mover) in &mut fast {
            if f == JUMP && fast.0 == 0 {
                mover.center += Vec3::new(40.0, 0.0, 25.0);
            }
            if f == GONE && fast.0 == 1 {
                commands.entity(entity).despawn();
            }
        }
        if let Ok(mut second) = second.single_mut() {
            let active = !(ONE_CAMERA..STAND_INS_OFF).contains(&f);
            if second.is_active != active {
                second.is_active = active;
            }
        }
        let scripted = |index: u32| {
            oracle
                .scripted
                .iter()
                .find(|(i, _)| *i == index)
                .map(|(_, entity)| *entity)
        };
        // A settled caster moves once (live again; its old place redraws).
        if f == CASTER_MOVE
            && let Some(entity) = scripted(1)
            && let Ok(mut transform) = casters.get_mut(entity)
        {
            transform.translation += Vec3::new(3.0, 0.0, -2.0);
        }
        // A caster appears after loading: it settles after the settle time.
        if f == CASTER_SPAWN {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(4.0, 9.0, 4.0))),
                MeshMaterial3d(materials.add(Color::srgb(0.4, 0.4, 0.9))),
                Transform::from_xyz(100.0, height(100.0, -80.0) + 4.5, -80.0),
            ));
        }
        // A settled caster goes.
        if f == CASTER_DESPAWN
            && let Some(entity) = scripted(4)
        {
            commands.entity(entity).despawn();
        }
        // Look reviews: the main view at fixed frames.
        if let Ok(dir) = std::env::var("SUN_CAPTURE") {
            let still = [110, 170, 300, 400, 860].contains(&f);
            let strip = (WALK + 60..WALK + 72).contains(&f) || (FLY + 20..FLY + 32).contains(&f);
            if still || strip {
                let kind = if still { "still" } else { "strip" };
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(format!("{dir}/{kind}-{f:04}.png")));
            }
        }
        if f >= EXIT_AT {
            report(&oracle, &counters, &mut exit);
        }
    }

    /// State checks, in the frame the lights' visibility ran.
    fn check_state(
        frame: Res<FrameCount>,
        mut oracle: ResMut<Oracle>,
        counters: Res<ShadowCacheCounters>,
        settings: Res<ShadowCacheSettings>,
        feedback: Res<ShadowCacheFeedback>,
        light: Query<(
            Entity,
            &Cascades,
            (&StableFarCascade, &StableFarProjections),
            &DirectionalLightShadowCache,
            &bevy::camera::visibility::CascadesVisibleEntities,
        )>,
        cameras: Query<(
            Entity,
            &GlobalTransform,
            &Projection,
            Has<MainCamera>,
            &VisibleEntities,
            &Camera,
        )>,
        representations: Query<&ShadowRepresentation>,
        faults: Res<ShadowCacheFaults>,
    ) {
        let f = frame.0;
        if !(CHECK_FROM..=CHECK_TO).contains(&f) {
            return;
        }
        let Ok((light_entity, cascades, (stable_far, projections), cache, visible)) =
            light.single()
        else {
            return;
        };
        let oracle = &mut *oracle;
        let mut fail = |what: String| oracle.failed.push((f, what));
        let mut checks = 0;
        // (An inactive camera has no cascades and draws nothing.)
        for (camera, transform, projection, main, camera_visible, _) in
            cameras.iter().filter(|(.., camera)| camera.is_active)
        {
            let Some(view_cascades) = cascades.cascades.get(&camera) else {
                fail(format!("camera {camera} has no cascades"));
                continue;
            };
            let far = view_cascades.last().unwrap();
            // Coverage: the far slice (with the blend overlap) in any
            // orientation is inside the far cascade's map.
            checks += 1;
            let world_from_view = transform.affine();
            let corners = projection.get_frustum_corners(-0.8 * (10.0 * 20f32.sqrt()), -200.0);
            let outside = corners.iter().any(|corner| {
                let clip =
                    far.clip_from_world * world_from_view.transform_point3a(*corner).extend(1.0);
                clip.x.abs() > 1.0 || clip.y.abs() > 1.0
            });
            if outside {
                fail(format!(
                    "camera {camera}: the far slice leaves the far cascade's map"
                ));
            }
            // The far projection survives a turn.
            if main {
                let rebuilds = projections.rebuilds(camera).unwrap_or(0);
                if (TURN + 1..WALK).contains(&f) {
                    checks += 1;
                    if oracle
                        .rebuilds
                        .last()
                        .is_some_and(|&(_, last)| last != rebuilds)
                    {
                        fail(format!(
                            "the far projection moved while the camera only turned ({rebuilds})"
                        ));
                    }
                }
                oracle.rebuilds.push((f, rebuilds));
            }
            // Stand-ins never reach a camera.
            checks += 1;
            if camera_visible
                .get(core::any::TypeId::of::<Mesh3d>())
                .iter()
                .any(|entity| {
                    matches!(
                        representations.get(*entity),
                        Ok(ShadowRepresentation::FarCascadeStandIn)
                    )
                })
            {
                fail(format!("camera {camera} sees a far stand-in"));
            }
            // The near cascades never draw stand-ins; the far one draws them
            // instead of the detail when the light selects them.
            if let Some(lists) = visible.entities.get(&camera) {
                checks += 1;
                let last = lists.len() - 1;
                for (index, list) in lists.iter().enumerate() {
                    for entity in &list.entities {
                        let representation = representations.get(*entity).ok();
                        let wrong = match representation {
                            Some(ShadowRepresentation::FarCascadeStandIn) => {
                                index != last || !stable_far.stand_ins
                            }
                            Some(ShadowRepresentation::Detail) => {
                                index == last && stable_far.stand_ins
                            }
                            None => false,
                        };
                        if wrong {
                            fail(format!(
                                "cascade {index} of camera {camera} draws {representation:?} {entity}"
                            ));
                            break;
                        }
                    }
                }
            }
        }
        // A view whose static revision isn't acknowledged keeps finding its
        // settled casters (the render world draws them only from such a
        // frame's list).
        let statuses = feedback.views();
        for (camera, view) in cache.views() {
            let id = ShadowViewId {
                light: light_entity,
                subview: ShadowSubview::Directional {
                    camera,
                    cascade: cache.cascade(),
                },
            };
            let acknowledged = statuses
                .get(&id)
                .filter(|status| status.generation == view.generation())
                .and_then(|status| status.acknowledged_static_revision);
            checks += 1;
            if !view.single_view()
                && acknowledged != Some(view.static_revision())
                && !view.statics_walked()
            {
                fail(format!(
                    "camera {camera}: revision {} unacknowledged ({acknowledged:?}) but its settled casters weren't walked",
                    view.static_revision()
                ));
            }
        }
        // A moving sun is never cached; a still one is, once settled.
        // (The sun first turns in the frame after SUN_MOVE.)
        let moving = (SUN_MOVE + 1..SUN_STOP).contains(&f);
        let settled = !(SUN_STEP..SUN_STEP + SETTLE + 2).contains(&f)
            && !(SUN_MOVE..SUN_STOP + SETTLE + 2).contains(&f);
        checks += 1;
        if moving && cache.is_cached() {
            fail("the sun is cached while it moves".into());
        }
        if settled && settings.enabled && !cache.is_cached() {
            fail("the still sun isn't cached".into());
        }
        // Camera LOD swaps never redraw the far cascade while it draws
        // stand-ins (only re-centers do, by projection, not by request).
        if settings.enabled && stable_far.stand_ins && (WALK + 1..CUT).contains(&f) {
            checks += 1;
            if oracle
                .walk_requests
                .is_some_and(|start| counters.rebuild_requests != start)
                && !faults.miss_static_invalidation
            {
                fail(format!(
                    "the walk's LOD swaps asked for redraws ({} since its start)",
                    counters.rebuild_requests - oracle.walk_requests.unwrap()
                ));
                oracle.walk_requests = Some(counters.rebuild_requests);
            }
        }
        oracle.checks += checks;
    }

    fn collect(results: Res<ShadowCacheAuditResults>, mut oracle: ResMut<Oracle>) {
        for record in results.take() {
            if !matches!(record.view.subview, ShadowSubview::Directional { .. }) {
                continue;
            }
            if !(CHECK_FROM..=CHECK_TO).contains(&record.frame) {
                continue;
            }
            match record.status {
                ShadowCacheAuditStatus::Missing(why) => warn!(
                    "oracle: frame {}: view {:?} ({:?}) not compared: {why}",
                    record.frame, record.view, record.plan
                ),
                _ if record.differing > 0 => warn!(
                    "oracle: frame {}: view {:?} ({:?}): {} texels differ, first at {:?}",
                    record.frame, record.view, record.plan, record.differing, record.first
                ),
                _ => {}
            }
            oracle.records.push((
                record.frame,
                record.view,
                record.status,
                record.differing,
                record.expected,
                format!("{:?}", record.plan),
            ));
        }
    }

    fn report(oracle: &Oracle, counters: &ShadowCacheCounters, exit: &mut MessageWriter<AppExit>) {
        let mut per_frame: HashMap<u32, (u32, u32)> = HashMap::default();
        let mut plans: HashMap<String, u32> = HashMap::default();
        for (frame, _, _, _, expected, plan) in &oracle.records {
            let entry = per_frame.entry(*frame).or_insert((0, *expected));
            entry.0 += 1;
            *plans.entry(plan.clone()).or_default() += 1;
        }
        // Frames without a record are allowed only while the sun is not
        // cached (it moved and hasn't settled).
        let unsettled = |frame: u32| {
            (SUN_STEP..SUN_STEP + SETTLE + 4).contains(&frame)
                || (SUN_MOVE..SUN_STOP + SETTLE + 4).contains(&frame)
        };
        let cache_on = env("SUN_ORACLE_CACHE").unwrap_or(1) != 0;
        let incomplete: Vec<u32> = (CHECK_FROM..=CHECK_TO)
            .filter(|frame| match per_frame.get(frame) {
                Some(&(found, expected)) => found != expected,
                None => cache_on && !unsettled(*frame),
            })
            .collect();
        let compared = oracle
            .records
            .iter()
            .filter(|record| record.2 == ShadowCacheAuditStatus::Compared)
            .count();
        let missing = oracle.records.len() - compared;
        let mut failing: Vec<u32> = oracle
            .records
            .iter()
            .filter(|record| {
                matches!(record.2, ShadowCacheAuditStatus::Missing(_))
                    || (record.2 == ShadowCacheAuditStatus::Compared && record.3 > 0)
            })
            .map(|record| record.0)
            .collect();
        let failing_count = failing.len();
        failing.sort_unstable();
        failing.dedup();
        for (frame, what) in oracle.failed.iter().take(20) {
            warn!("oracle: frame {frame}: {what}");
        }
        if let Some(first) = incomplete.first() {
            warn!(
                "oracle: {} frames without a record for every cached view, first {first}",
                incomplete.len()
            );
        }
        let pass = failing_count == 0 && incomplete.is_empty() && oracle.failed.is_empty();
        let mut plans: Vec<_> = plans.into_iter().collect();
        plans.sort_unstable();
        info!(
            "oracle: {}: {} view-frames ({compared} compared, {missing} missing), {failing_count} failing in {} frames{}; {} incomplete frames; {} of {} state checks failed; plans {plans:?}; {:?}",
            if pass { "PASS" } else { "FAIL" },
            oracle.records.len(),
            failing.len(),
            failing
                .first()
                .map(|frame| format!(", first at frame {frame}"))
                .unwrap_or_default(),
            incomplete.len(),
            oracle.failed.len(),
            oracle.checks,
            *counters,
        );
        exit.write(if pass {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}
