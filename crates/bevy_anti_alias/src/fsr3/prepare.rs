use super::Fsr3;
use bevy_camera::{Camera3d, CameraMainTextureUsages, MainPassResolutionOverride};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_math::{ops, UVec2, Vec2, Vec4Swizzles};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, Extent3d, Texture, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};
use std::sync::Mutex;
use wgpu_ffx::{FsrContext, FsrContextFlags, FsrContextInfo, FsrView};

/// A camera's FSR 3 state: the compiled pipelines, recreated only when the context flags change, and the
/// resolution-dependent resources, reallocated when the render or output resolution changes.
#[derive(Component)]
pub struct Fsr3RenderContext {
    pub context: FsrContext,
    pub flags: FsrContextFlags,
    pub view: Mutex<Fsr3ViewResources>,
}

/// The resources of one camera at one pair of render and output resolutions.
pub struct Fsr3ViewResources {
    pub view: FsrView,
    pub render_resolution: UVec2,
    pub upscaled_resolution: UVec2,
    /// FSR 3 outputs, at the render resolution, that later effects could reuse.
    pub dilated_depth: Texture,
    pub dilated_motion_vectors: Texture,
    pub reconstructed_previous_depth: Buffer,
    /// Discard the history at the next dispatch: set by a reset request or a frame FSR 3 did not run on.
    pub reset_history: bool,
}

impl Fsr3ViewResources {
    fn new(
        context: &FsrContext,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        render_resolution: UVec2,
        upscaled_resolution: UVec2,
    ) -> Self {
        let view = context.create_view(
            render_queue,
            render_resolution.to_array(),
            upscaled_resolution.to_array(),
        );
        let render_texture = |label, format| {
            render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: render_resolution.x,
                    height: render_resolution.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        Self {
            view,
            render_resolution,
            upscaled_resolution,
            dilated_depth: render_texture("fsr3_dilated_depth", TextureFormat::R32Float),
            dilated_motion_vectors: render_texture(
                "fsr3_dilated_motion_vectors",
                TextureFormat::Rg16Float,
            ),
            reconstructed_previous_depth: render_device.create_buffer(&BufferDescriptor {
                label: Some("fsr3_reconstructed_previous_depth"),
                size: render_resolution.element_product() as u64 * size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            reset_history: true,
        }
    }
}

pub fn prepare_fsr3(
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &Fsr3,
            &mut Camera3d,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            &mut MipBias,
            Option<&mut Fsr3RenderContext>,
            Option<&MainPassResolutionOverride>,
        ),
        (
            With<Camera3d>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (
        entity,
        view,
        fsr3,
        mut camera_3d,
        mut camera_main_texture_usages,
        mut temporal_jitter,
        mut mip_bias,
        fsr3_context,
        resolution_override,
    ) in &mut query
    {
        camera_main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;

        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();

        let upscaled_resolution = view.viewport.zw();
        // `extract_fsr3` passes only views whose render resolution FSR 3 accepts.
        let render_resolution = fsr3.quality_mode.render_resolution(upscaled_resolution);

        // Bevy's color is HDR and its depth reversed (1.0 at the near plane). Bevy's perspective projection has an
        // infinite far plane; a custom one may not.
        let mut flags = FsrContextFlags::HIGH_DYNAMIC_RANGE | FsrContextFlags::DEPTH_INVERTED;
        if view.clip_from_view.z_axis.z == 0.0 {
            flags |= FsrContextFlags::DEPTH_INFINITE;
        }

        let phase_count = wgpu_ffx::get_jitter_phase_count(
            render_resolution.x as i32,
            upscaled_resolution.x as i32,
        );
        let jitter =
            wgpu_ffx::get_jitter_offset((frame_count.0 % phase_count as u32) as i32, phase_count);
        // `TemporalJitter` moves the image the opposite way from FSR's jitter offset of the same sign, see the node.
        temporal_jitter.offset = -Vec2::from(jitter);
        // MipBias is extracted from the main world every frame, so it is set every frame. This is FidelityFX's
        // recommended bias, log2(render / output) - 1.
        mip_bias.0 = ops::log2(render_resolution.x as f32 / upscaled_resolution.x as f32) - 1.0;

        match fsr3_context {
            Some(fsr3_context) if fsr3_context.flags.bits() == flags.bits() => {
                let mut resources = fsr3_context.view.lock().unwrap();
                if resources.render_resolution != render_resolution
                    || resources.upscaled_resolution != upscaled_resolution
                {
                    *resources = Fsr3ViewResources::new(
                        &fsr3_context.context,
                        &render_device,
                        &render_queue,
                        render_resolution,
                        upscaled_resolution,
                    );
                }
                // `extract_fsr3` removes the override on frames FSR 3 does not run.
                if resolution_override.map(|o| o.0) != Some(render_resolution) {
                    commands
                        .entity(entity)
                        .insert(MainPassResolutionOverride(render_resolution));
                }
            }
            _ => {
                let context = FsrContext::new(FsrContextInfo {
                    device: render_device.wgpu_device().clone(),
                    flags,
                });
                let resources = Fsr3ViewResources::new(
                    &context,
                    &render_device,
                    &render_queue,
                    render_resolution,
                    upscaled_resolution,
                );
                commands.entity(entity).insert((
                    Fsr3RenderContext {
                        context,
                        flags,
                        view: Mutex::new(resources),
                    },
                    MainPassResolutionOverride(render_resolution),
                ));
            }
        }
    }
}
