//! Point lights whose cube shadow maps are cached face by face
//! ([`ShadowCacheSettings`]): each face keeps its settled casters in a static
//! layer of its own and draws only its live casters every frame.
//!
//! The scene: a floor with walls and pillars, four shadowed point lights (one
//! quiet, one with movers crossing its face edges and corners, one that moves
//! for a while, one that is hidden and shown, which renumbers the others'
//! layers), and a fifth light spawned and removed later.
//!
//! `--oracle` runs a script and checks every face of every shadowed point
//! light against a full redraw by a twin (`ShadowCacheAuditPlugin`), texel by
//! texel, plus state checks: a still light is cached and its quiet faces are
//! kept, a moving light is never cached, a face whose static revision isn't
//! acknowledged walks its settled casters, the faces with static layers fit
//! the budget, a mover crossing an edge is in both faces. Environment:
//! `SHADOW_CACHE_FAULT=NAME` injects a fault the oracle must catch;
//! `POINT_ORACLE_CACHE=0` runs with the cache off; `POINT_ORACLE_BUDGET_MIB`
//! sets the static layers' budget (256; a 1024 face takes 4 MiB);
//! `POINT_ORACLE_RESTORE_NOT_READY=1` keeps the restore's blit counted as not
//! compiled (every face falls back); `POINT_ORACLE_READINESS_DELAY=n` holds a
//! mesh's shadow pipeline back for its first n frames (15);
//! `POINT_ORACLE_INVALIDATE_EVERY=n` redraws every static layer every n
//! frames; `POINT_ORACLE_NO_CPU_CULLING=1` culls every mesh on the GPU only;
//! `POINT_ORACLE_HASHES=FILE` runs without twins and writes a hash of every
//! face per frame, for a comparison with a run with the cache off.

use std::f32::consts::*;

use bevy::{
    asset::uuid_handle,
    camera::visibility::{NoCpuCulling, NoFrustumCulling},
    diagnostic::FrameCount,
    light::{
        InvalidateShadowCache, PointLightShadowCache, PointLightShadowMap, ShadowCacheCounters,
        ShadowCacheFaults, ShadowCacheFeedback, ShadowCacheLoading, ShadowCacheSettings,
        ShadowCacheState, ShadowSubview, ShadowViewId,
    },
    math::ops,
    mesh::MeshTag,
    pbr::{
        ShadowCacheAuditPlugin, ShadowCacheAuditResults, ShadowCacheAuditSettings,
        ShadowCacheAuditStatus,
    },
    platform::collections::HashMap,
    prelude::*,
    reflect::TypePath,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
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
const CHECK_TO: u32 = 800;
const EXIT_AT: u32 = 820;

/// The script (first frame of each event).
const QUIET: (u32, u32) = (100, 150);
const CASTER_MOVE: u32 = 160;
const RETAG: u32 = 200;
const CASTER_SPAWN: u32 = 240;
const CASTER_DESPAWN: u32 = 280;
const HIDE: u32 = 320;
const SHOW: u32 = 360;
const LIGHT_MOVE: u32 = 400;
const LIGHT_STOP: u32 = 440;
/// The camera looks away (every light dormant); meanwhile the hidden light
/// goes and comes back, so the others' layers move and return.
const AWAY: u32 = 480;
const AWAY_HIDE: u32 = 500;
const AWAY_SHOW: u32 = 520;
const BACK: u32 = 540;
const LIGHT_SPAWN: u32 = 580;
const JUMP: u32 = 620;
const GONE: u32 = 650;
const LIGHT_DESPAWN: u32 = 680;
/// The camera looks away again, with no light renumbered meanwhile: on its
/// return each light restores the tiles its movers were drawn into.
const AWAY2: u32 = 690;
const BACK2: u32 = 710;
const QUIET2: (u32, u32) = (720, 790);

/// The lights' places.
const QUIET_LIGHT: Vec3 = Vec3::new(-10.0, 5.0, -10.0);
const BUSY_LIGHT: Vec3 = Vec3::new(10.0, 3.0, 10.0);
const MOVING_LIGHT: Vec3 = Vec3::new(10.0, 5.0, -10.0);
const HIDDEN_LIGHT: Vec3 = Vec3::new(-10.0, 4.0, 10.0);
const NEW_LIGHT: Vec3 = Vec3::new(0.0, 6.0, 0.0);

fn main() -> AppExit {
    let oracle = std::env::args().any(|arg| arg == "--oracle");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "shadow cache: point lights".into(),
            resolution: WindowResolution::new(640, 360),
            present_mode: PresentMode::AutoVsync,
            ..default()
        }),
        ..default()
    }))
    .insert_resource(PointLightShadowMap { size: 1024 })
    .insert_resource(ShadowCacheSettings {
        enabled: env("POINT_ORACLE_CACHE").unwrap_or(1) != 0,
        budget_bytes: env::<u64>("POINT_ORACLE_BUDGET_MIB").unwrap_or(256) << 20,
        settle_time: FRAME * SETTLE,
    })
    .add_plugins(MaterialPlugin::<TagShadowMaterial>::default())
    .add_systems(Startup, setup)
    .add_systems(Update, move_movers);
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            TAG_SHADOW_SHADER.id(),
            Shader::from_wesl(TAG_SHADOW_SHADER_SOURCE, "point_tag_shadow.wesl"),
        )
        .unwrap();
    if env("POINT_ORACLE_NO_CPU_CULLING").unwrap_or(0) != 0 {
        app.add_observer(|add: On<Add<Mesh3d>>, mut commands: Commands| {
            commands.entity(add.entity).insert(NoCpuCulling);
        });
    }
    if oracle {
        oracle::add(&mut app);
    }
    app.run()
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// A material whose shadow depends on the mesh's [`MeshTag`]: a tag other
/// than 0 draws the mesh at half its size (inside its bounds).
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
struct TagShadowMaterial {}

const TAG_SHADOW_SHADER: Handle<Shader> = uuid_handle!("0b7d3c1e-5f2a-4e8b-9c61-7a4d2e9f1b53");

const TAG_SHADOW_SHADER_SOURCE: &str = r"
import bevy_pbr::render::{mesh_functions, view_transformations::position_world_to_clip};

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let scale = select(1.0, 0.5, mesh_functions::get_tag(vertex.instance_index) != 0u);
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position * scale, 1.0),
    );
    var out: VertexOutput;
    out.position = position_world_to_clip(world_position.xyz);
    return out;
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.8, 0.7, 0.3, 1.0);
}
";

impl Material for TagShadowMaterial {
    fn vertex_shader() -> ShaderRef {
        TAG_SHADOW_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        TAG_SHADOW_SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        TAG_SHADOW_SHADER.into()
    }
}

/// A light of the script, by name.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
struct OracleLight(&'static str);

/// A caster of the script, by name.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
struct Scripted(&'static str);

/// Moves every frame, by simulation time.
#[derive(Component, Clone, Copy)]
enum Mover {
    /// Around `center` at `radius` in the horizontal plane.
    Circle {
        center: Vec3,
        radius: f32,
        speed: f32,
    },
    /// Along `axis` through `center`, `amplitude` each way.
    Line {
        center: Vec3,
        axis: Vec3,
        amplitude: f32,
        speed: f32,
    },
}

fn point_light(range: f32) -> PointLight {
    PointLight {
        intensity: 400_000.0,
        range,
        shadow_maps_enabled: true,
        ..default()
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut tag_materials: ResMut<Assets<TagShadowMaterial>>,
) {
    let stone = materials.add(Color::srgb(0.7, 0.7, 0.75));
    let moving = materials.add(Color::srgb(1.0, 0.7, 0.3));

    // The floor and walls.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(60.0, 0.5, 60.0))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(0.0, -0.25, 0.0),
    ));
    let wall = meshes.add(Cuboid::new(30.0, 8.0, 0.5));
    for (x, z, turn) in [(0.0, -20.0, 0.0), (-20.0, 0.0, FRAC_PI_2), (8.0, 20.0, 0.0)] {
        commands.spawn((
            Mesh3d(wall.clone()),
            MeshMaterial3d(stone.clone()),
            Transform::from_xyz(x, 4.0, z).with_rotation(Quat::from_rotation_y(turn)),
        ));
    }
    // Pillars and boxes around every light, some on face edges and corners.
    let pillar = meshes.add(Cuboid::new(0.8, 6.0, 0.8));
    let block = meshes.add(Cuboid::new(1.5, 1.0, 2.5));
    for (light_index, light) in [QUIET_LIGHT, BUSY_LIGHT, MOVING_LIGHT, HIDDEN_LIGHT]
        .into_iter()
        .enumerate()
    {
        for index in 0..7u32 {
            let angle = index as f32 * 2.399 + light_index as f32;
            let distance = 3.0 + index as f32 * 0.9;
            let (x, z) = (distance * ops::cos(angle), distance * ops::sin(angle));
            let (mesh, y) = if index % 2 == 0 {
                (pillar.clone(), 3.0)
            } else {
                (block.clone(), 0.5 + index as f32 * 0.6)
            };
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(stone.clone()),
                Transform::from_xyz(light.x + x, y, light.z + z)
                    .with_rotation(Quat::from_rotation_y(angle)),
            ));
        }
    }
    // The quiet light's scripted casters: one on the corner of its +X, +Y
    // and world -Z faces (moved once), one whose shadow its mesh tag decides
    // (retagged), one that goes.
    let cube = meshes.add(Cuboid::from_length(1.0));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(stone.clone()),
        Transform::from_translation(QUIET_LIGHT + Vec3::new(3.0, 3.0, -3.0)),
        Scripted("corner"),
    ));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(tag_materials.add(TagShadowMaterial {})),
        MeshTag::new(0),
        Transform::from_translation(QUIET_LIGHT + Vec3::new(-3.0, -2.0, 0.5)),
        Scripted("tagged"),
    ));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(stone.clone()),
        Transform::from_translation(QUIET_LIGHT + Vec3::new(0.5, -1.5, 3.5)),
        Scripted("goes"),
    ));

    // Movers around the busy light: one circling at its height (across the
    // side faces' edges), one bobbing through its top face's edges, one
    // passing right by it (across the near plane of several faces), two fast
    // ones (one jumps, one goes).
    let mover = meshes.add(Cuboid::new(0.8, 0.8, 0.8));
    let movers = [
        (
            "circle",
            Mover::Circle {
                center: BUSY_LIGHT,
                radius: 4.0,
                speed: 0.8,
            },
        ),
        (
            "bob",
            Mover::Line {
                center: BUSY_LIGHT + Vec3::new(1.5, 2.0, 0.3),
                axis: Vec3::Y,
                amplitude: 3.0,
                speed: 1.1,
            },
        ),
        (
            "by",
            Mover::Line {
                center: BUSY_LIGHT + Vec3::new(0.0, 0.8, 0.3),
                axis: Vec3::X,
                amplitude: 6.0,
                speed: 0.5,
            },
        ),
        (
            "fast0",
            Mover::Circle {
                center: BUSY_LIGHT + Vec3::new(0.0, -1.5, 0.0),
                radius: 8.0,
                speed: 0.75,
            },
        ),
        (
            "fast1",
            Mover::Circle {
                center: BUSY_LIGHT + Vec3::new(1.0, -1.0, -1.0),
                radius: 6.0,
                speed: -0.9,
            },
        ),
    ];
    for (name, motion) in movers {
        commands.spawn((
            Mesh3d(mover.clone()),
            MeshMaterial3d(moving.clone()),
            Transform::default(),
            motion,
            Scripted(name),
        ));
    }

    // The lights. The hidden light is spawned first so that it sorts first:
    // hiding it renumbers the others' layers. Every shadowed light is kept
    // unculled (as MPS does while the cache is on), so a light out of view is
    // planned dormant, not dropped.
    for (name, at, range) in [
        ("hidden", HIDDEN_LIGHT, 10.0),
        ("quiet", QUIET_LIGHT, 12.0),
        ("busy", BUSY_LIGHT, 12.0),
        ("moving", MOVING_LIGHT, 11.0),
    ] {
        commands.spawn((
            point_light(range),
            Transform::from_translation(at),
            NoFrustumCulling,
            OracleLight(name),
        ));
    }

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 22.0, 34.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn move_movers(time: Res<Time>, mut movers: Query<(&Mover, &mut Transform)>) {
    let t = time.elapsed_secs();
    for (mover, mut transform) in &mut movers {
        transform.translation = match *mover {
            Mover::Circle {
                center,
                radius,
                speed,
            } => {
                center
                    + Vec3::new(
                        radius * ops::cos(t * speed),
                        0.0,
                        radius * ops::sin(t * speed),
                    )
            }
            Mover::Line {
                center,
                axis,
                amplitude,
                speed,
            } => center + axis * amplitude * ops::sin(t * speed),
        };
        transform.rotation = Quat::from_rotation_y(t * 0.7);
    }
}

mod oracle {
    use super::*;
    use std::io::Write;

    /// Whether every light lights no view in a frame, by the script alone:
    /// the camera looks away (every light's range is behind it).
    fn all_dormant(frame: u32) -> bool {
        (AWAY..BACK).contains(&frame) || (AWAY2..BACK2).contains(&frame)
    }

    /// What the oracle collects.
    #[derive(Resource, Default)]
    struct Oracle {
        /// (frame, view, status, differing, expected, plan).
        records: Vec<(u32, ShadowViewId, ShadowCacheAuditStatus, u32, u32, String)>,
        /// With `POINT_ORACLE_HASHES`: (frame, light, face, hash, cached, plan).
        hashes: Vec<(u32, &'static str, u8, String, bool, String)>,
        /// Frames whose state checks failed, with what.
        failed: Vec<(u32, String)>,
        checks: u32,
        /// Frames in which the circling mover was in two faces of its light.
        edge_frames: u32,
        /// The lights' names by entity (kept after a light goes).
        names: HashMap<Entity, &'static str>,
        /// Face-frames whose dormancy disagrees with the script: dormant
        /// while the camera sees the light, or a cached face drawn while it
        /// looks away.
        wrong_dormancy: Vec<(u32, ShadowViewId, String)>,
    }

    fn hashes_file() -> Option<String> {
        std::env::var("POINT_ORACLE_HASHES").ok()
    }

    fn cache_on() -> bool {
        env("POINT_ORACLE_CACHE").unwrap_or(1) != 0
    }

    pub fn add(app: &mut App) {
        let fault = std::env::var("SHADOW_CACHE_FAULT").unwrap_or_default();
        let faults = ShadowCacheFaults {
            skip_restore: fault == "skip_restore",
            drop_written: fault == "drop_written",
            drop_one_tile: fault == "drop_one_tile",
            miss_static_invalidation: fault == "miss_static_invalidation",
            miss_settle_invalidation: fault == "miss_settle_invalidation",
            miss_tag_change: fault == "miss_tag_change",
            late_marker: fault == "late_marker",
            stale_ack: fault == "stale_ack",
            wrong_layer: fault == "wrong_layer",
            acknowledge_early: fault == "acknowledge_early",
            skip_reentry_restore: fault == "skip_reentry_restore",
            wrong_face: fault == "wrong_face",
            miss_face_edge: fault == "miss_face_edge",
            miss_face_invalidation: fault == "miss_face_invalidation",
            miss_index_change: fault == "miss_index_change",
            swap_destination: fault == "swap_destination",
            false_dormant: fault == "false_dormant",
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
            restore_not_ready: env("POINT_ORACLE_RESTORE_NOT_READY").unwrap_or(0) != 0,
            readiness_delay_frames: env("POINT_ORACLE_READINESS_DELAY").unwrap_or(15),
            ..faults
        };
        let hashes = hashes_file().is_some();
        app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
            .insert_resource(faults)
            .insert_resource(ShadowCacheAuditSettings {
                enabled: true,
                fresh_redraws: !hashes,
                hashes,
            })
            .add_plugins(ShadowCacheAuditPlugin)
            .init_resource::<Oracle>()
            .add_systems(Update, script.before(move_movers))
            .add_systems(
                PostUpdate,
                check_state.after(bevy::light::SimulationLightSystems::CheckLightVisibility),
            )
            .add_systems(Last, collect);
    }

    #[expect(clippy::too_many_arguments, reason = "a script")]
    fn script(
        frame: Res<FrameCount>,
        mut oracle: ResMut<Oracle>,
        mut commands: Commands,
        mut loading: ResMut<ShadowCacheLoading>,
        mut camera: Query<&mut Transform, With<Camera3d>>,
        mut lights: Query<
            (Entity, &OracleLight, &mut Transform, &mut Visibility),
            Without<Camera3d>,
        >,
        mut casters: Query<
            (Entity, &Scripted, &mut Transform, Option<&mut Mover>),
            (Without<Camera3d>, Without<OracleLight>),
        >,
        mut tags: Query<&mut MeshTag>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut invalidations: MessageWriter<InvalidateShadowCache>,
        counters: Res<ShadowCacheCounters>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let f = frame.0;
        if f == LOADED_AT {
            loading.0 = false;
        }
        if let Some(every) = env::<u32>("POINT_ORACLE_INVALIDATE_EVERY")
            && every > 0
            && f.is_multiple_of(every)
        {
            invalidations.write(InvalidateShadowCache::All);
        }
        // The camera: over the room, or looking away (every light dormant).
        if let Ok(mut transform) = camera.single_mut() {
            let wanted = if (AWAY..BACK).contains(&f) || (AWAY2..BACK2).contains(&f) {
                Transform::from_xyz(0.0, 6.0, 40.0).looking_at(Vec3::new(0.0, 6.0, 100.0), Vec3::Y)
            } else {
                Transform::from_xyz(0.0, 22.0, 34.0).looking_at(Vec3::ZERO, Vec3::Y)
            };
            if *transform != wanted {
                *transform = wanted;
            }
        }
        for (entity, light, mut transform, mut visibility) in &mut lights {
            oracle.names.insert(entity, light.0);
            match light.0 {
                "hidden" => {
                    let hidden = (HIDE..SHOW).contains(&f) || (AWAY_HIDE..AWAY_SHOW).contains(&f);
                    let wanted = if hidden {
                        Visibility::Hidden
                    } else {
                        Visibility::Inherited
                    };
                    if *visibility != wanted {
                        *visibility = wanted;
                    }
                }
                // Moves along a line, then stays where it stopped.
                "moving" => {
                    let t = f.clamp(LIGHT_MOVE, LIGHT_STOP) - LIGHT_MOVE;
                    let at = MOVING_LIGHT + Vec3::new(-0.05, 0.02, 0.04) * t as f32;
                    if transform.translation != at {
                        transform.translation = at;
                    }
                }
                "new" if f == LIGHT_DESPAWN => commands.entity(entity).despawn(),
                _ => {}
            }
        }
        if f == LIGHT_SPAWN {
            commands.spawn((
                point_light(9.0),
                Transform::from_translation(NEW_LIGHT),
                NoFrustumCulling,
                OracleLight("new"),
            ));
        }
        for (entity, scripted, mut transform, mover) in &mut casters {
            match scripted.0 {
                // A settled caster across three faces moves once: live again,
                // every face its old place was in redraws.
                "corner" if f == CASTER_MOVE => transform.translation += Vec3::new(0.6, -0.4, 0.3),
                "tagged" if f == RETAG => {
                    if let Ok(mut tag) = tags.get_mut(entity) {
                        *tag = MeshTag::new(1);
                    }
                }
                "goes" if f == CASTER_DESPAWN => commands.entity(entity).despawn(),
                "fast0" if f == JUMP => {
                    if let Some(mut mover) = mover
                        && let Mover::Circle { center, .. } = &mut *mover
                    {
                        *center += Vec3::new(-3.0, 0.5, 2.0);
                    }
                }
                "fast1" if f == GONE => commands.entity(entity).despawn(),
                _ => {}
            }
        }
        // A caster appears after loading: it settles after the settle time.
        if f == CASTER_SPAWN {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.0, 2.0, 1.0))),
                MeshMaterial3d(materials.add(Color::srgb(0.4, 0.4, 0.9))),
                Transform::from_translation(QUIET_LIGHT + Vec3::new(-2.5, -2.0, -2.5)),
            ));
        }
        if f >= EXIT_AT {
            report(&oracle, &counters, &mut exit);
        }
    }

    /// State checks, in the frame the lights' visibility ran.
    fn check_state(
        frame: Res<FrameCount>,
        mut oracle: ResMut<Oracle>,
        settings: Res<ShadowCacheSettings>,
        shadow_map: Res<PointLightShadowMap>,
        feedback: Res<ShadowCacheFeedback>,
        faults: Res<ShadowCacheFaults>,
        lights: Query<(
            Entity,
            &OracleLight,
            &PointLightShadowCache,
            &bevy::camera::visibility::CubemapVisibleEntities,
            &ViewVisibility,
        )>,
        casters: Query<(Entity, &Scripted)>,
    ) {
        let f = frame.0;
        if !(CHECK_FROM..=CHECK_TO).contains(&f) {
            return;
        }
        let oracle = &mut *oracle;
        let mut fail = |what: String| oracle.failed.push((f, what));
        let mut checks = 0;
        let statuses = feedback.views();
        // (Redrawn on purpose, or GPU-culled movers anywhere in the world,
        // which cover every face whole: no face is kept.)
        let invalidate_every = env::<u32>("POINT_ORACLE_INVALIDATE_EVERY").is_some_and(|n| n > 0)
            || env("POINT_ORACLE_NO_CPU_CULLING").unwrap_or(0) != 0;
        let status_of = |light: Entity, face: usize, cache: &PointLightShadowCache| {
            statuses
                .get(&ShadowViewId {
                    light,
                    subview: ShadowSubview::PointFace(face as u8),
                })
                .filter(|status| status.generation == cache.face(face).generation())
        };
        let circle = casters
            .iter()
            .find(|(_, scripted)| scripted.0 == "circle")
            .map(|(entity, _)| entity);
        let mut with_layer = 0u64;
        for (light, name, cache, lists, visibility) in &lights {
            if !visibility.get() {
                continue;
            }
            // A still light is cached once settled; a moving one never.
            checks += 1;
            let settled = match name.0 {
                "moving" => !(LIGHT_MOVE..LIGHT_STOP + SETTLE + 2).contains(&f),
                "new" => f >= LIGHT_SPAWN + 3,
                _ => true,
            };
            let moving = name.0 == "moving" && (LIGHT_MOVE + 1..LIGHT_STOP).contains(&f);
            if moving && cache.is_cached() {
                fail(format!("the {} light is cached while it moves", name.0));
            }
            if settled && settings.enabled && !cache.is_cached() {
                fail(format!("the still {} light isn't cached", name.0));
            }
            // The circling mover crosses its light's face edges (lists
            // found on the CPU, cached or not).
            if name.0 == "busy"
                && let Some(circle) = circle
                && (0..6)
                    .filter(|face| lists.get(*face).entities.contains(&circle))
                    .count()
                    >= 2
            {
                oracle.edge_frames += 1;
            }
            if !cache.is_cached() {
                continue;
            }
            for face in 0..6 {
                let view = cache.face(face);
                let status = status_of(light, face, cache);
                // A face whose revision isn't acknowledged keeps finding its
                // settled casters (its static layer is drawn only from such a
                // frame's list).
                checks += 1;
                let acknowledged = status.and_then(|status| status.acknowledged_static_revision);
                if view.lights_a_view()
                    && !view.single_view()
                    && acknowledged != Some(view.static_revision())
                    && !view.statics_walked()
                {
                    fail(format!(
                        "{} face {face}: revision {} unacknowledged ({acknowledged:?}) but its settled casters weren't walked",
                        name.0,
                        view.static_revision()
                    ));
                }
                if status.is_some_and(|status| !status.fallback) {
                    with_layer += 1;
                }
                // The quiet light's faces are kept while nothing happens.
                let quiet = (QUIET.0..QUIET.1).contains(&f) || (QUIET2.0..QUIET2.1).contains(&f);
                if name.0 == "quiet" && quiet && !invalidate_every {
                    checks += 1;
                    match status {
                        Some(status) if status.state == ShadowCacheState::Kept => {}
                        other => fail(format!(
                            "quiet face {face} isn't kept: {:?}",
                            other.map(|status| status.state)
                        )),
                    }
                }
            }
        }
        // The faces with a usable static layer fit the budget (a face whose
        // layer was given up counts for a few frames more, so this bounds
        // the layers in use from above).
        checks += 1;
        let face_bytes = (shadow_map.size as u64).pow(2) * 4;
        if with_layer * face_bytes > settings.budget_bytes && !faults.restore_not_ready {
            fail(format!(
                "{with_layer} faces with static layers exceed the budget of {} MiB",
                settings.budget_bytes >> 20
            ));
        }
        oracle.checks += checks;
    }

    fn collect(results: Res<ShadowCacheAuditResults>, mut oracle: ResMut<Oracle>) {
        let hashes = hashes_file().is_some();
        for record in results.take() {
            let ShadowSubview::PointFace(face) = record.view.subview else {
                continue;
            };
            if !(CHECK_FROM..=CHECK_TO).contains(&record.frame) {
                continue;
            }
            let plan = format!("{:?}", record.plan);
            // Dormancy checked apart from the cache: a dormant face is never
            // compared, so the cache must not call one dormant that a view
            // can see.
            let dormant = record.status == ShadowCacheAuditStatus::Dormant;
            // (A record's frame is the frame count extracted after the
            // frame's `Last`: one more than the script saw in that frame.)
            let away = all_dormant(record.frame.wrapping_sub(1));
            if dormant != (away && record.cached && cache_on()) {
                oracle
                    .wrong_dormancy
                    .push((record.frame, record.view, plan.clone()));
            }
            if hashes {
                let hash = match (record.status, record.hash) {
                    (ShadowCacheAuditStatus::Compared, Some(hash)) => format!("{hash:016x}"),
                    (ShadowCacheAuditStatus::Dormant, _) => "dormant".to_string(),
                    _ => "missing".to_string(),
                };
                let name = oracle.names.get(&record.light).copied().unwrap_or("?");
                oracle
                    .hashes
                    .push((record.frame, name, face, hash, record.cached, plan));
                continue;
            }
            match record.status {
                ShadowCacheAuditStatus::Missing(why) => warn!(
                    "oracle: frame {}: view {:?} ({:?}) not compared: {why}",
                    record.frame, record.view, record.plan
                ),
                _ if record.differing > 0 => warn!(
                    "oracle: frame {}: view {:?} (layers {:?}, {:?}, revision {} static {:?} working {:?}): {} texels differ, first at {:?}",
                    record.frame,
                    record.view,
                    record.layers,
                    record.plan,
                    record.static_revision,
                    record.static_content,
                    record.working,
                    record.differing,
                    record.first
                ),
                _ => {}
            }
            oracle.records.push((
                record.frame,
                record.view,
                record.status,
                record.differing,
                record.expected,
                plan,
            ));
        }
    }

    fn report(oracle: &Oracle, counters: &ShadowCacheCounters, exit: &mut MessageWriter<AppExit>) {
        if let Some(path) = hashes_file() {
            let mut hashes = oracle.hashes.clone();
            hashes.sort_unstable();
            // Never over an earlier run's file.
            let mut file = std::fs::File::create_new(&path)
                .unwrap_or_else(|error| panic!("the hashes file {path}: {error}"));
            for (frame, light, face, hash, cached, plan) in &hashes {
                writeln!(
                    file,
                    "{frame} {light} {face} {hash} {} {plan}",
                    u8::from(*cached)
                )
                .unwrap();
            }
            let cached = hashes.iter().filter(|hash| hash.4).count();
            for (frame, view, plan) in oracle.wrong_dormancy.iter().take(10) {
                warn!(
                    "oracle: frame {frame}: {view:?} dormancy disagrees with the script ({plan})"
                );
            }
            info!(
                "oracle: HASHES: {} face-frames ({cached} cached) written to {path}; {} of {} state checks failed; {:?}",
                hashes.len(),
                oracle.failed.len(),
                oracle.checks,
                *counters
            );
            for (frame, what) in oracle.failed.iter().take(20) {
                warn!("oracle: frame {frame}: {what}");
            }
            exit.write(
                if oracle.failed.is_empty() && oracle.wrong_dormancy.is_empty() {
                    AppExit::Success
                } else {
                    AppExit::error()
                },
            );
            return;
        }
        let mut per_frame: HashMap<u32, (u32, u32)> = HashMap::default();
        let mut plans: HashMap<String, u32> = HashMap::default();
        for (frame, _, _, _, expected, plan) in &oracle.records {
            let entry = per_frame.entry(*frame).or_insert((0, *expected));
            entry.0 += 1;
            *plans.entry(plan.clone()).or_default() += 1;
        }
        // Every checked frame has a record for each face of each shadowed
        // light (the audit's count of what it expected).
        let incomplete: Vec<u32> = (CHECK_FROM..=CHECK_TO)
            .filter(|frame| {
                per_frame
                    .get(frame)
                    .is_none_or(|&(found, expected)| found != expected)
            })
            .collect();
        let compared = oracle
            .records
            .iter()
            .filter(|record| record.2 == ShadowCacheAuditStatus::Compared)
            .count();
        let dormant = oracle
            .records
            .iter()
            .filter(|record| record.2 == ShadowCacheAuditStatus::Dormant)
            .count();
        let missing = oracle.records.len() - compared - dormant;
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
                "oracle: {} frames without a record for every face, first {first}",
                incomplete.len()
            );
        }
        let mut failed_checks = oracle.failed.len() + oracle.wrong_dormancy.len();
        for (frame, view, plan) in oracle.wrong_dormancy.iter().take(10) {
            warn!("oracle: frame {frame}: {view:?} dormancy disagrees with the script ({plan})");
        }
        // (Meshes culled on the GPU only aren't in the CPU's lists.)
        if oracle.edge_frames == 0 && env("POINT_ORACLE_NO_CPU_CULLING").unwrap_or(0) == 0 {
            warn!("oracle: the circling mover never crossed a face edge");
            failed_checks += 1;
        }
        let cached_plans = cache_on() as usize
            * plans
                .iter()
                .filter(|(plan, _)| plan.as_str() != "Fallback")
                .map(|(_, count)| *count as usize)
                .sum::<usize>();
        if cache_on() && cached_plans == 0 {
            warn!("oracle: no face was ever planned by the cache");
            failed_checks += 1;
        }
        let pass = failing_count == 0 && incomplete.is_empty() && failed_checks == 0;
        let mut plans: Vec<_> = plans.into_iter().collect();
        plans.sort_unstable();
        info!(
            "oracle: {}: {} face-frames ({compared} compared, {dormant} dormant, {missing} missing), {failing_count} failing in {} frames{}; {} incomplete frames; {failed_checks} of {} state checks failed; {} edge frames; plans {plans:?}; {:?}",
            if pass { "PASS" } else { "FAIL" },
            oracle.records.len(),
            failing.len(),
            failing
                .first()
                .map(|frame| format!(", first at frame {frame}"))
                .unwrap_or_default(),
            incomplete.len(),
            oracle.checks,
            oracle.edge_frames,
            *counters,
        );
        exit.write(if pass {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}
