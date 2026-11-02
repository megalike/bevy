use super::{prepare::Fsr3RenderContext, Fsr3};
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_ecs::{
    query::{Has, With},
    system::{Commands, Query, ResMut},
};
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_fsr3(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    cleanup_query: Query<Has<Fsr3>>,
) {
    let mut cameras_3d = main_world
        .query_filtered::<(RenderEntity, &Camera, &Projection, Option<&mut Fsr3>), With<Hdr>>();

    for (entity, camera, camera_projection, mut fsr3) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(fsr3) = fsr3.as_mut()
            && camera.is_active
            && camera_projection.is_perspective()
        {
            entity_commands.insert(Fsr3::clone(fsr3));
            fsr3.reset = false;
        } else if cleanup_query.get(entity) == Ok(true) {
            entity_commands.remove::<(Fsr3, Fsr3RenderContext, MainPassResolutionOverride)>();
        }
    }
}
