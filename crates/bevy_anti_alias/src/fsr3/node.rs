use super::{prepare::Fsr3RenderContext, Fsr3};
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::system::Res;
use bevy_math::{ops, Vec4Swizzles};
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    diagnostic::RecordDiagnostics,
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewTarget},
};
use bevy_time::{Real, Time};
use wgpu_ffx::{FsrDispatchFlags, FsrDispatchInfo};

pub fn fsr3(
    view: ViewQuery<(
        &Fsr3,
        &Fsr3RenderContext,
        &MainPassResolutionOverride,
        &TemporalJitter,
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &ViewPrepassTextures,
    )>,
    time: Res<Time<Real>>,
    mut ctx: RenderContext,
) {
    let (
        fsr3,
        fsr3_context,
        resolution_override,
        temporal_jitter,
        extracted_view,
        extracted_camera,
        view_target,
        prepass_textures,
    ) = view.into_inner();

    let mut resources = fsr3_context.view.lock().unwrap();
    let render_resolution = resolution_override.0;
    let upscaled_resolution = extracted_view.viewport.zw();
    let (Some(prepass_depth_texture), Some(prepass_motion_vectors_texture)) =
        (&prepass_textures.depth, &prepass_textures.motion_vectors)
    else {
        resources.reset_history = true;
        return;
    };
    if resources.render_resolution != render_resolution
        || resources.upscaled_resolution != upscaled_resolution
    {
        // `prepare_fsr3` sets both together; a mismatch would make FSR read outside its resources.
        resources.reset_history = true;
        return;
    }

    // Reversed-Z perspective: view depth z maps to (a z + b) / -z, which is 1.0 at the near plane and 0.0 at the
    // far plane (a = 0 for an infinite far plane).
    let clip_from_view = extracted_view.clip_from_view;
    let (a, b) = (clip_from_view.z_axis.z, clip_from_view.w_axis.z);
    let near = b / (1.0 + a);
    let far = if a == 0.0 { f32::INFINITY } else { b / a };
    let fov_y = 2.0 * ops::atan(1.0 / clip_from_view.y_axis.y);

    let view_target = view_target.post_process_write();

    let dispatch_info = FsrDispatchInfo {
        color: (**view_target.source_texture).clone(),
        depth: (*prepass_depth_texture.texture.texture).clone(),
        motion_vectors: (*prepass_motion_vectors_texture.texture.texture).clone(),
        exposure: None,
        reactive_mask: None,
        transparency_and_composition: None,
        dilated_depth: (*resources.dilated_depth).clone(),
        dilated_motion_vectors: (*resources.dilated_motion_vectors).clone(),
        reconstructed_previous_depth: (*resources.reconstructed_previous_depth).clone(),
        output: (**view_target.destination_texture).clone(),
        // FSR's jitter offset of +x moves the image by +x pixels, Bevy's `TemporalJitter` by -x.
        jitter_offset: (-temporal_jitter.offset).to_array(),
        // Bevy's motion vectors are the current minus the previous position in UV units; FSR wants the offset
        // from the current to the previous position in pixels of the render resolution.
        motion_vector_scale: (-render_resolution.as_vec2()).to_array(),
        render_size: render_resolution.to_array(),
        upscale_size: upscaled_resolution.to_array(),
        enable_sharpening: fsr3.enable_sharpening,
        sharpness: fsr3.sharpness.clamp(0.0, 1.0),
        // FSR refuses a frame time below 1 ms (the first frame reports 0).
        frame_time_delta: (time.delta_secs() * 1000.0).max(1.0),
        // The main pass multiplies the color by the camera's exposure: that is FSR's pre-exposure, which scales the
        // history when it changes. Its `exposure` input stays the default 1.0, since the color FSR sees is already
        // the exposed signal that tonemapping receives.
        pre_exposure: extracted_camera.exposure,
        reset_history: resources.reset_history,
        // With reversed depth FSR expects its "near" value to be the larger distance.
        camera_near: far,
        camera_far: near,
        camera_fov_y: fov_y,
        view_space_to_meters_factor: 1.0,
        flags: FsrDispatchFlags::empty(),
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "fsr3");

    fsr3_context
        .context
        .dispatch(&mut resources.view, ctx.command_encoder(), &dispatch_info)
        .expect("Failed to dispatch FSR 3");
    resources.reset_history = false;

    time_span.end(ctx.command_encoder());
}
