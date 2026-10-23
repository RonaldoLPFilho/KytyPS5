#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_

#include "common/common.h"

#include <memory>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
struct ImageInfo;
struct WindowContext;

class Presenter final {
public:
	struct Frame;

	explicit Presenter(WindowContext& window);
	~Presenter();
	KYTY_CLASS_NO_COPY(Presenter);

	// `hdr`: the guest buffer holds BT.2100 PQ pixels (shown on an HDR10 swapchain when there is one).
	[[nodiscard]] Frame&         PrepareFrame(CommandBuffer& command, const ImageInfo& info,
	                                         bool hdr = false);
	[[nodiscard]] Frame&         PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
	                                               CommandBuffer* producer = nullptr);
	[[nodiscard]] Frame*         PrepareLastFrame();
	// The vblank thread, when no flip was presented: shows the frame a present turned away (it came within a display
	// refresh of the one before) if nothing newer replaced it and the display can take it now.
	bool                         PresentSkippedIfDue();
	[[nodiscard]] bool           IsGuestPaused() const noexcept;
	// --hdr and the display takes HDR10: the game is told it may output HDR.
	[[nodiscard]] bool           IsHdrSupported() const noexcept;
	[[nodiscard]] bool           NeedsSystemOverlayRefresh() const noexcept;
	[[nodiscard]] RenderContext& Renderer() const noexcept;
	// KYTY_FLIP_RATE caps the frame's flip. With frame generation only frames carrying its inputs
	// are capped (the game's 3D view): movies and menus keep a flip per vblank.
	[[nodiscard]] static bool FlipRateApplies(const Frame& frame) noexcept;
	void                         Present(Frame& frame, bool reuse = false);
	void                         Discard(Frame& frame);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
