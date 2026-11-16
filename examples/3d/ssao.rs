//! A scene showcasing screen space ambient occlusion.

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::Hdr,
    math::ops,
    pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel},
    prelude::*,
    render::camera::TemporalJitter,
};
use std::f32::consts::PI;

fn main() {
    App::new()
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        near_view(),
        Hdr,
        Msaa::Off,
        ScreenSpaceAmbientOcclusion::default(),
        TemporalAntiAliasing::default(),
    ));

    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.5),
        perceptual_roughness: 1.0,
        reflectance: 0.0,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(0.0, 0.0, 1.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(0.0, -1.0, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(1.0, 0.0, 0.0),
    ));
    // A large ground plane, to show SSAO on distant surfaces (F switches to a far view).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4000.0, 4000.0))),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, -1.5, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.4).mesh().uv(72, 36))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.4, 0.4),
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        })),
        SphereMarker,
    ));

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, PI * -0.15, PI * -0.15)),
    ));

    commands.spawn((
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn near_view() -> Transform {
    Transform::from_xyz(-2.0, 2.0, -2.0).looking_at(Vec3::ZERO, Vec3::Y)
}

fn far_view() -> Transform {
    Transform::from_xyz(-200.0, 150.0, -200.0).looking_at(Vec3::new(200.0, -1.5, 200.0), Vec3::Y)
}

fn update(
    camera: Single<
        (
            Entity,
            &mut Transform,
            Option<&ScreenSpaceAmbientOcclusion>,
            Option<&TemporalJitter>,
        ),
        (With<Camera>, Without<SphereMarker>),
    >,
    mut text: Single<&mut Text>,
    mut sphere: Single<&mut Transform, With<SphereMarker>>,
    mut commands: Commands,
    keycode: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    sphere.translation.y = ops::sin(time.elapsed_secs() / 1.7) * 0.7;

    let (camera_entity, mut camera_transform, ssao, temporal_jitter) = camera.into_inner();
    let current_ssao = ssao.cloned().unwrap_or_default();

    let mut commands = commands.entity(camera_entity);
    commands
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Low,
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::Digit2),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::Digit3),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::Digit4),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::Digit5),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                constant_object_thickness: (current_ssao.constant_object_thickness * 2.0).min(4.0),
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::ArrowUp),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                constant_object_thickness: (current_ssao.constant_object_thickness * 0.5)
                    .max(0.0625),
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::ArrowDown),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                radius: (current_ssao.radius * 1.25).min(8.0),
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::ArrowRight),
        )
        .insert_if(
            ScreenSpaceAmbientOcclusion {
                radius: (current_ssao.radius * 0.8).max(0.1),
                ..current_ssao
            },
            || keycode.just_pressed(KeyCode::ArrowLeft),
        );
    if keycode.just_pressed(KeyCode::KeyF) {
        *camera_transform = if *camera_transform == near_view() {
            far_view()
        } else {
            near_view()
        };
    }
    if keycode.just_pressed(KeyCode::Digit1) {
        commands.remove::<ScreenSpaceAmbientOcclusion>();
    }
    if keycode.just_pressed(KeyCode::Space) {
        if temporal_jitter.is_some() {
            commands.remove::<TemporalJitter>();
        } else {
            commands.insert(TemporalJitter::default());
        }
    }

    text.clear();

    let (o, l, m, h, u) = match ssao.map(|s| s.quality_level) {
        None => ("*", "", "", "", ""),
        Some(ScreenSpaceAmbientOcclusionQualityLevel::Low) => ("", "*", "", "", ""),
        Some(ScreenSpaceAmbientOcclusionQualityLevel::Medium) => ("", "", "*", "", ""),
        Some(ScreenSpaceAmbientOcclusionQualityLevel::High) => ("", "", "", "*", ""),
        Some(ScreenSpaceAmbientOcclusionQualityLevel::Ultra) => ("", "", "", "", "*"),
        _ => unreachable!(),
    };

    if let Some(radius) = ssao.map(|s| s.radius) {
        text.push_str(&format!("Radius: {radius} (Left/Right)\n"));
    }

    if let Some(thickness) = ssao.map(|s| s.constant_object_thickness) {
        text.push_str(&format!(
            "Constant object thickness: {thickness} (Up/Down)\n\n"
        ));
    }

    text.push_str("SSAO Quality:\n");
    text.push_str(&format!("(1) {o}Off{o}\n"));
    text.push_str(&format!("(2) {l}Low{l}\n"));
    text.push_str(&format!("(3) {m}Medium{m}\n"));
    text.push_str(&format!("(4) {h}High{h}\n"));
    text.push_str(&format!("(5) {u}Ultra{u}\n\n"));

    text.push_str("Temporal Antialiasing:\n");
    text.push_str(match temporal_jitter {
        Some(_) => "(Space) Enabled",
        None => "(Space) Disabled",
    });

    text.push_str("\n\n(F) Near / far view");
}

#[derive(Component)]
struct SphereMarker;
