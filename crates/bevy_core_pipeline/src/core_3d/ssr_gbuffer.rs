//! A packed G-buffer written by the forward opaque pass, for screen-space reflections on forward-rendered views.
//!
//! MPS: a forward material that is too expensive to evaluate twice cannot feed Bevy's deferred prepass; instead the
//! main opaque pass writes the packed surface data as a second color target while it shades. [`ForwardSsrGBuffer`]
//! on a camera asks for it; [`ViewSsrGBuffer`] is the render-world texture; the main opaque pass node attaches it
//! as slot 1, cleared to zero once per frame (zero means "no receiver"), and the mesh pipelines of the `Opaque3d`
//! and `AlphaMask3d` phases gain the matching target (`MeshPipelineKey::FORWARD_SSR_MRT` in `bevy_pbr`). Every
//! shader drawn into the pass either writes valid receiver data or zero; the skybox writes nothing (an empty write
//! mask), so the background keeps the clear.

use bevy_camera::Camera3d;
use bevy_color::LinearRgba;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryItem, With},
    reflect::ReflectComponent,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut},
};
use bevy_image::ToExtents;
use bevy_log::warn_once;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::ExtractComponent,
    render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::RenderDevice,
    sync_component::SyncComponent,
    texture::{ColorAttachment, TextureCache},
    view::Msaa,
    RenderApp,
};

/// The format of the forward SSR G-buffer: Bevy's deferred packing (`deferred/types.wesl`), one `vec4<u32>` per
/// pixel.
pub const FORWARD_SSR_GBUFFER_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

/// Add this component to a 3D camera to make its main opaque pass write the packed G-buffer that forward-mode
/// screen-space reflections read.
///
/// The camera needs [`Msaa::Off`]: the packed format cannot be multisampled, so a multisampled view renders without
/// the buffer (a warning is logged once).
#[derive(Component, Default, Reflect, Clone, Copy)]
#[reflect(Component, Default, Clone)]
pub struct ForwardSsrGBuffer;

impl ForwardSsrGBuffer {
    /// Whether a view with this marker and the given MSAA setting gets the buffer.
    pub fn is_supported(msaa: Msaa) -> bool {
        msaa.samples() == 1
    }
}

impl SyncComponent<RenderApp> for ForwardSsrGBuffer {
    type Target = (Self, ViewSsrGBuffer);
}

impl ExtractComponent<RenderApp> for ForwardSsrGBuffer {
    type QueryData = Read<ForwardSsrGBuffer>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(_item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(ForwardSsrGBuffer)
    }
}

/// The packed G-buffer of a view with [`ForwardSsrGBuffer`]: the size of the view's render target, cleared to zero
/// when the main opaque pass first attaches it each frame.
#[derive(Component)]
pub struct ViewSsrGBuffer {
    pub texture: ColorAttachment,
    pub size: Extent3d,
}

/// Creates the G-buffer texture of every view that asks for it (re-created through the texture cache when the
/// target's size changes) and removes it from views that no longer do.
pub fn prepare_ssr_gbuffer_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &Msaa, Has<ForwardSsrGBuffer>), With<Camera3d>>,
) {
    for (entity, camera, msaa, wants_gbuffer) in &views {
        let Some(physical_target_size) = camera.physical_target_size.filter(|_| wants_gbuffer)
        else {
            commands.entity(entity).remove::<ViewSsrGBuffer>();
            continue;
        };
        if !ForwardSsrGBuffer::is_supported(*msaa) {
            warn_once!(
                "ForwardSsrGBuffer needs Msaa::Off; the view {entity} renders without the buffer"
            );
            commands.entity(entity).remove::<ViewSsrGBuffer>();
            continue;
        }
        let size = physical_target_size.to_extents();
        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("forward_ssr_gbuffer_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: FORWARD_SSR_GBUFFER_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(ViewSsrGBuffer {
            texture: ColorAttachment::new(texture, None, None, Some(LinearRgba::BLACK.into())),
            size,
        });
    }
}
