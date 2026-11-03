use super::{prepare::Fsr3RenderContext, Fsr3};
use crate::taa::TemporalAntiAliasing;
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_ecs::{
    entity::Entity,
    query::{Has, With},
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::UVec2;
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    sync_world::RenderEntity,
    view::Msaa,
    Extract, MainWorld,
};
use bevy_time::{Real, Time};
use bevy_utils::once;
use tracing::warn;

/// The smallest render resolution, per axis, FSR 3 runs at: its luminance pyramid has six mip levels at half the
/// render resolution.
const MIN_RENDER_RESOLUTION: u32 = 64;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
type OtherTemporalResolver = bevy_ecs::query::Or<(
    With<TemporalAntiAliasing>,
    With<crate::dlss::Dlss<crate::dlss::DlssSuperResolutionFeature>>,
    With<crate::dlss::Dlss<crate::dlss::DlssRayReconstructionFeature>>,
)>;
#[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
type OtherTemporalResolver = With<TemporalAntiAliasing>;

pub fn extract_fsr3(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    render_query: Query<(Has<Fsr3>, Option<&Fsr3RenderContext>)>,
) {
    // A camera resolves its history with one temporal technique; FSR 3 yields to the others.
    let conflicting: Vec<Entity> = main_world
        .query_filtered::<Entity, (With<Fsr3>, OtherTemporalResolver)>()
        .iter(&main_world)
        .collect();
    if !conflicting.is_empty() {
        once!(warn!(
            "A camera has `Fsr3` together with `TemporalAntiAliasing` or `Dlss`; FSR 3 is skipped on it"
        ));
    }

    let mut cameras_3d = main_world.query::<(
        Entity,
        RenderEntity,
        &Camera,
        Option<&Projection>,
        Option<&mut Fsr3>,
        Has<Hdr>,
        Option<&Msaa>,
        Has<DepthPrepass>,
        Has<MotionVectorPrepass>,
        (Has<TemporalJitter>, Has<MipBias>),
    )>();

    for (
        main_entity,
        entity,
        camera,
        projection,
        fsr3,
        hdr,
        msaa,
        depth_prepass,
        motion_vector_prepass,
        (temporal_jitter, mip_bias),
    ) in cameras_3d.iter_mut(&mut main_world)
    {
        let Ok((had_fsr3, fsr3_context)) = render_query.get(entity) else {
            continue;
        };
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");

        let Some(mut fsr3) = fsr3.filter(|_| camera.is_active) else {
            if had_fsr3 || fsr3_context.is_some() {
                entity_commands.remove::<(Fsr3, Fsr3RenderContext, MainPassResolutionOverride)>();
            }
            continue;
        };

        let eligible = hdr
            && projection.is_some_and(Projection::is_perspective)
            && msaa.copied().unwrap_or_default() == Msaa::Off
            && depth_prepass
            && motion_vector_prepass
            && temporal_jitter
            && mip_bias
            && !conflicting.contains(&main_entity)
            // A minimized window has a zero size.
            && camera.physical_viewport_size().is_some_and(|size| {
                let render_resolution = fsr3.quality_mode.render_resolution(size);
                render_resolution.cmpge(UVec2::splat(MIN_RENDER_RESOLUTION)).all()
            });

        if eligible {
            entity_commands.insert(Fsr3::clone(&fsr3));
        } else {
            // Render at the native resolution without jitter until the camera is eligible again. The context and
            // its pipelines stay; the history they hold does not.
            entity_commands.remove::<(Fsr3, MainPassResolutionOverride)>();
        }

        // A reset request, and a skipped frame, stay latched in the render world until a dispatch consumes them.
        if (fsr3.reset || !eligible)
            && let Some(fsr3_context) = fsr3_context
        {
            fsr3_context.view.lock().unwrap().reset_history = true;
        }
        if fsr3.reset {
            fsr3.reset = false;
        }
    }
}

/// FSR 3 wants the real duration of the last frame; the render world has only the virtual clock.
pub fn extract_real_time(mut commands: Commands, time: Extract<Res<Time<Real>>>) {
    commands.insert_resource(**time);
}
