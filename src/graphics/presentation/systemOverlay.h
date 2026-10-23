#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <memory>
#include <string>

union SDL_Event;

namespace Libs::Graphics {

struct GraphicContext;

struct SystemOverlayVisualState {
	bool     active;
	uint64_t revision;
};

// Small panels over the game image, drawn with every frame (dialog or not): the frame rate
// (KYTY_FPS_HUD), the progress of background work (the shader prefetch) and the debug warp.
struct SystemOverlayHud {
	std::string title;  // large first line (empty: no frame rate panel)
	std::string detail; // small second line
	std::string status; // the progress line (empty: none), over a bar
	float       status_fraction = 0.0f;
	std::string notice; // a text panel (empty: none)
	vk::Rect2D  region; // the game image in the swapchain image
	vk::Rect2D  drawn;  // out: the panels, in swapchain pixels
};

void                     InitializeSystemOverlayInput();
void                     ShutdownSystemOverlayInput();
bool                     ProcessSystemOverlayInput(const SDL_Event& event);
SystemOverlayVisualState GetSystemOverlayVisualState() noexcept;

class SystemOverlay final {
public:
	explicit SystemOverlay(GraphicContext& graphics);
	~SystemOverlay();
	KYTY_CLASS_NO_COPY(SystemOverlay);

	[[nodiscard]] bool PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t image_count,
	                                SystemOverlayHud* hud = nullptr);
	void               Record(vk::CommandBuffer command, vk::ImageView target);
	void               ReleaseVulkan();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
