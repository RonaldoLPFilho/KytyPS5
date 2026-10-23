#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_H_
// DLSS Frame Generation through NVIDIA Streamline (KYTY_FRAMEGEN=<generated frames, 1..3>), for
// Demon's Souls PPSA01341 01.007.000 and 01.005.000. The inputs come from the game's own passes, found by frame
// capture (docs/RE-DEMONS-SOULS.md): the G-buffer vertex shaders' constants hold the camera
// (world->clip matrix, the previous frame's, and the TAA jitter), the TAA dispatch reads the depth
// and motion-vector images. The renderer thread gathers them per guest frame; at the guest flip
// they are copied next to the presenter's frame; the present thread hands them to Streamline and
// presents through it (sl.interposer.dll is the process' Vulkan loader, see Initialize()).
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace Libs::Graphics {
class CommandBuffer;
class TextureCache;
namespace FrameGen {

// The TAA compute shader of 01.007.000 and of 01.005.000 (the same shader recompiled).
constexpr std::array<uint64_t, 2> TaaShaders {0xa44e580f4f0cfb2bULL, 0x934a6e1197b99148ULL};
inline bool IsTaaShader(uint64_t hash) {
	return std::find(TaaShaders.begin(), TaaShaders.end(), hash) != TaaShaders.end();
}
struct CameraShaderLayout {
	uint64_t hash;
	uint32_t view_proj_offset;
	uint32_t prev_view_proj_offset;
	uint32_t jitter_offset;
};
// Captured from Demon's Souls 01.007.000 (and the last three from 01.005.000). Several vertex shaders expose the
// same camera data after different SRT prefixes; offsets are the start of the row-major world-to-clip matrix.
// Shaders not listed here are learned while playing (frame-gen.cpp, LearnedLayout).
constexpr std::array<CameraShaderLayout, 22> CameraShaders {{
	{0xf4d2e78cd1769fc7ULL, 29, 45, 57}, {0x08e16923875aa8e9ULL, 29, 45, 57},
	{0x0ffcbf08be26e74eULL, 29, 45, 57}, {0xa49f5c52d5cb9a32ULL, 29, 45, 57},
	{0x96fc6004d7988611ULL, 30, 46, 58}, {0x798287b1ae24bfd7ULL, 32, 50, 48},
	{0x2c0e577a01705282ULL, 33, 49, 61}, {0x1c6f994c079d5ebcULL, 33, 49, 61},
	{0x224c6adc8751dfe6ULL, 33, 49, 61}, {0x6828ffbb94cee94cULL, 33, 49, 61},
	{0x32448d81a089ef4aULL, 33, 49, 61}, {0x1b33cf5a16357f4bULL, 33, 49, 61},
	{0xec57eea9f55ffc93ULL, 33, 49, 61},
	{0xf73f6b14c9773b02ULL, 33, 49, 61}, {0xf9bf6001b44aab5bULL, 37, 53, 65},
	{0x0bd18e197f768a69ULL, 37, 53, 65}, {0x4ded98185f00d165ULL, 38, 54, 66},
	{0xbf2adfba2013424dULL, 41, 57, 69},
	// Captured in the Nexus: current/previous camera matrices and TAA jitter follow a 4-word prefix.
	{0xeabb515b7eb79db5ULL, 33, 49, 61},
	// 01.005.000's G-buffer vertex shaders (upstream's list).
	{0x1d69b5d27198446aULL, 29, 45, 57}, {0x5931e753b757f3bcULL, 29, 45, 57},
	{0xa97f610824942162ULL, 29, 45, 57},
}};

struct Camera {
	std::array<float, 16> view_proj {};
	std::array<float, 16> prev_view_proj {};
	std::array<float, 2>  jitter_ndc {};
	bool                  valid = false;
};

// Where a frame's camera came from: a listed shader, a learned one, or (neither drew this frame) the last camera
// held with no camera motion (the motion vectors carry it all).
enum class CameraSource : uint8_t { Table, Learned, Held };

// One presented frame's inputs, owned by the presenter's frame.
struct FrameInputs {
	VulkanImage   depth, motion;
	vk::ImageView depth_view = nullptr, motion_view = nullptr;
	vk::Extent2D  render_extent {};
	Camera        camera;
	CameraSource  camera_source = CameraSource::Table;
	uint64_t      guest_frame = 0;
	bool          motion_attachment = false; // the motion copy is a color attachment too
	bool          valid       = false;
};

// Process start, before the window: loads Streamline when KYTY_FRAMEGEN is set (it then becomes
// SDL's Vulkan loader, and so the proxy of instance, device and swapchain calls).
void Initialize();
// After Initialize: Streamline is loaded (device creation then adds what it relies on but does
// not add itself, VK_KHR_present_id for its VK_NV_low_latency2).
[[nodiscard]] bool Requested();
// After the device exists: checks that DLSS-G is available on it.
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);
// Right after the dispatcher is loaded for the device (before the recording layer copies it):
// the command-buffer calls Streamline intercepts only for its command-list state tracking
// (disabled here) go straight to the driver.
void BypassCommandHooks(vk::Device device);
[[nodiscard]] bool Enabled();
// True when the present-thread Streamline lifecycle must run, including while F2 has disabled FG.
[[nodiscard]] bool PresentIntegrationReady();
// True when renderer inputs are needed by DLSS-G.
[[nodiscard]] bool CaptureRequested();

// Present thread, around vkQueuePresentKHR of the image showing `inputs` (null: a frame
// without inputs, e.g. a menu). `backbuffer` is the swapchain extent, `region` the part the
// game image covers.
void BeforePresent(const FrameInputs* inputs, vk::Extent2D color_extent, vk::Extent2D backbuffer,
                   vk::Rect2D region, vk::Format format, uint32_t image_count, bool reusing_last);
void AfterPresent();
// Present thread: the frames DLSS-G showed per presented frame (1 without generation), as last
// queried (about once a second), for the KYTY_FPS_HUD panel.
[[nodiscard]] uint32_t PresentedFrames();
[[nodiscard]] bool ToggleFrameGeneration();
[[nodiscard]] bool FpsHudEnabled();
[[nodiscard]] bool ToggleFpsHud();
[[nodiscard]] const char* RuntimeStatus();
// KYTY_FPS_HUD: the panel's rectangle, normalized to the game image. Its motion vectors are
// zeroed so generated frames keep the panel in place.
void SetStaticRect(float x0, float y0, float x1, float y1);

// Renderer thread (Kyty.Gpu).
void OnDraw(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);
void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address);
// At the guest flip, in the frame's command buffer: copy the gathered images into `inputs`.
void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs);
void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_H_
