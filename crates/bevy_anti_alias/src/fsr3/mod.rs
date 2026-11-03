//! AMD `FidelityFX` Super Resolution 3 (FSR 3) upscaler.
//!
//! FSR 3 is a temporal upscaler: it renders the main pass at a lower resolution, jitters the camera each frame
//! and reconstructs a full-resolution, anti-aliased image from the current frame and the reprojected history.
//! Its [`Fsr3QualityMode::NativeAa`] mode renders at full resolution and uses it as temporal anti-aliasing only.
//!
//! Works on any GPU with compute shaders; uses [`wgpu_ffx`], which runs AMD's precompiled SPIR-V shaders.
//!
//! # Usage
//! 1. Enable Bevy's `fsr3` feature
//! 2. Add the [`Fsr3`] component to your camera entity, optionally setting a specific [`Fsr3QualityMode`] (defaults to `Quality`)
//! 3. Custom rendering code, including third party crates, should account for the optional `MainPassResolutionOverride` to work with FSR 3 (see the `custom_render_phase` example)

mod extract;
mod node;
mod prepare;

use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderSystems,
};

/// Enables FSR 3 for cameras with the [`Fsr3`] component.
#[derive(Default)]
pub struct Fsr3Plugin;

impl Plugin for Fsr3Plugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                ExtractSchedule,
                (extract::extract_fsr3, extract::extract_real_time),
            )
            .add_systems(
                Render,
                prepare::prepare_fsr3
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            )
            .add_systems(Core3d, node::fsr3.in_set(Core3dSystems::EarlyPostProcess));
    }
}

/// Camera component to enable FSR 3.
///
/// FSR 3 runs while the camera is active and has [`Hdr`], a perspective projection,
/// [`Msaa::Off`](bevy_render::view::Msaa::Off), the depth and motion vector prepasses, and a viewport large
/// enough for the quality mode. On other frames, such as while its window is minimized, the camera renders at
/// its native resolution without jitter, and FSR 3 starts again with a fresh history.
///
/// FSR 3 is skipped, with a warning, on a camera that also has
/// [`TemporalAntiAliasing`](crate::taa::TemporalAntiAliasing) or `Dlss`: a camera uses one temporal upscaler or
/// anti-aliasing technique at a time.
///
/// FSR 3 needs correct motion vectors for everything on screen, like
/// [`TemporalAntiAliasing`](crate::taa::TemporalAntiAliasing).
///
/// # Limitations
/// - The viewport must start at the render target's origin: a viewport with a nonzero position, as in split
///   screen, is not supported.
/// - Motion blur samples the prepass textures at the output resolution, so it does not match the reduced render
///   resolution.
/// - `wgpu_ffx` creates its staging buffers anew for every frame (its bind groups are kept per set of resources).
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct Fsr3 {
    /// How much upscaling should be applied.
    pub quality_mode: Fsr3QualityMode,
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
    /// Run FSR 3's robust contrast adaptive sharpening (RCAS) pass on the upscaled image. Set to false to turn
    /// sharpening off.
    pub enable_sharpening: bool,
    /// Sharpening strength, from 0.0 (none) to 1.0 (most). Used only with `enable_sharpening`.
    pub sharpness: f32,
}

impl Default for Fsr3 {
    fn default() -> Self {
        Self {
            quality_mode: Fsr3QualityMode::default(),
            reset: false,
            enable_sharpening: true,
            sharpness: 0.8,
        }
    }
}

/// How much FSR 3 upscales: the ratio of the output resolution to the main pass's render resolution.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Default)]
#[reflect(Default, Clone, PartialEq)]
pub enum Fsr3QualityMode {
    /// No upscaling (1.0x): anti-aliasing only.
    NativeAa,
    /// 1.5x upscaling per axis.
    #[default]
    Quality,
    /// 1.7x upscaling per axis.
    Balanced,
    /// 2.0x upscaling per axis.
    Performance,
    /// 3.0x upscaling per axis.
    UltraPerformance,
}

impl Fsr3QualityMode {
    /// The ratio of the output resolution to the render resolution, per axis.
    pub fn upscale_ratio(self) -> f32 {
        match self {
            Self::NativeAa => 1.0,
            Self::Quality => 1.5,
            Self::Balanced => 1.7,
            Self::Performance => 2.0,
            Self::UltraPerformance => 3.0,
        }
    }

    /// The render resolution for an output resolution, rounded down like the `FidelityFX` SDK.
    pub fn render_resolution(self, upscaled_resolution: UVec2) -> UVec2 {
        (upscaled_resolution.as_vec2() / self.upscale_ratio()).as_uvec2()
    }
}
