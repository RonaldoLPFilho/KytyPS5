#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"
#include "loader/demonsSoulsWarp.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif
#include "frame-gen.h"
#include "local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string_view>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <optional>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

#define KYTY_ENABLE_DEBUG_PRINTF
#define KYTY_DBG_INPUT

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage            image;
	FrameGen::FrameInputs  fg; // frame generation inputs (KYTY_FRAMEGEN)
	uint64_t               present_tick = 0;
	bool                   busy         = false;
	bool                   reusing_last = false;
	bool                   hdr          = false; // BT.2100 PQ pixels
	bool                   blank        = false; // a cleared frame (no guest picture)

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
			FrameGen::DestroyInputs(m_window.graphic_ctx, frame->fg);
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		if (count == 0 || format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires at least one frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!m_frames.empty()) {
			EXIT("prepared-frame pool was initialized twice\n");
		}
		m_format = format;
		m_frames.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		if (format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires a presentation format\n");
		}
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		if (m_format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool has no presentation format\n");
		}
		return m_format;
	}

	Presenter::Frame* Acquire() {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		while (m_free.empty()) {
			m_available.Wait(&m_mutex);
		}
		auto* frame = m_free.front();
		m_free.pop_front();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		if (m_last_frame == frame) {
			m_last_frame = nullptr;
		}
		frame->busy         = true;
		frame->reusing_last = false;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	Presenter::Frame* AcquireLast() {
		m_mutex.Lock();
		auto* frame = m_last_frame;
		if (frame == nullptr) {
			m_mutex.Unlock();
			return nullptr;
		}
		auto free = std::find(m_free.begin(), m_free.end(), frame);
		if (free == m_free.end() || frame->busy) {
			m_mutex.Unlock();
			EXIT("last submitted frame is not available for reuse\n");
		}
		m_free.erase(free);
		m_last_frame        = nullptr;
		frame->busy         = true;
		frame->reusing_last = true;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame, bool reuse) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy || frame->reusing_last != reuse) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame, bool make_last = false) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy         = false;
		frame->reusing_last = false;
		if (make_last) {
			m_last_frame = frame;
		}
		m_free.push_back(frame);
		m_available.Signal();
	}

private:
	void WaitForFrame(Presenter::Frame& frame) { m_scheduler.Wait(frame.present_tick); }

	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	Common::CondVar                                m_available;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::deque<Presenter::Frame*>                  m_free;
	Presenter::Frame*                              m_last_frame = nullptr;
	vk::Format                                     m_format     = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eTransferSrc | vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}
	if (dst.image != nullptr) {
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
	image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost, Timeout }; // desport: S-001 Timeout

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	[[nodiscard]] bool   Create(); // desport: S-001 false: no swapchain (0x0 surface)
	[[nodiscard]] bool   Valid() const noexcept { return m_handle != nullptr; } // desport: S-001
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] Status AcquireNextImage();
	[[nodiscard]] bool   PrepareSystemOverlay(SystemOverlayHud* hud = nullptr);
	void                 RecordPresentCommands(CommandBuffer& command, VulkanImage& source,
	                                           bool draw_system_overlay);
	uint64_t             Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] vk::Extent2D Extent() const noexcept { return m_extent; }
	[[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(std::memory_order_relaxed); }
	// The guest's frames are PQ (true) or SDR; true when the swapchain must be recreated for it.
	[[nodiscard]] bool SetHdr(bool hdr) {
		m_hdr_requested = hdr;
		return m_hdr_active != (hdr && IsHdrSupported());
	}
	// Where RecordPresentCommands put the frame in the swapchain image.
	[[nodiscard]] vk::Rect2D PresentRegion() const noexcept { return m_present_region; }

private:
	void                 Destroy();
	vk::SurfaceFormatKHR SelectFormat();

	WindowContext&              m_window;
	vk::SwapchainKHR            m_handle = nullptr;
	vk::Format                  m_format = vk::Format::eUndefined;
	std::atomic_bool            m_hdr_supported {false};
	bool                        m_hdr_requested = false;
	bool                        m_hdr_active    = false;
	vk::Extent2D                m_extent {};
	std::vector<vk::Image>      m_images;
	std::vector<vk::ImageView>  m_image_views;
	std::vector<vk::Semaphore>  m_image_acquired;
	std::vector<vk::Semaphore>  m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	uint32_t                    m_image_index = static_cast<uint32_t>(-1);
	uint32_t                    m_frame_index = 0;
	vk::Rect2D                  m_present_region {};
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		EXIT_IF(!swapchain.Create()); // desport: S-001
		// One frame more than the swapchain's images: with frame generation the present worker holds
		// the frame it presents and the next one waiting, while the GPU thread prepares a third.
		frames.Initialize(swapchain.ImageCount() + 1u, swapchain.Format());
	}
	~Impl() { StopAsync(); }
	KYTY_CLASS_NO_COPY(Impl);

	// Presents the frame on the calling thread (the vblank thread, or the present worker).
	void PresentNow(Frame& frame, bool reuse);

	// KYTY_FRAMEGEN: DLSS-G's vkQueuePresentKHR shows the generated frame, paces, then shows the rendered one, and
	// can take more than a 60 Hz vblank. On the vblank thread that held up the virtual vblank and the guest's flips
	// (the game ran at 30 fps with frame generation on, 50-60 without). While frame generation is on, the vblank
	// thread hands the frame to this worker and goes on; it waits only when a frame is already waiting.
	// KYTY_FRAMEGEN_SYNC_PRESENT=1 presents on the vblank thread as before (for comparison).
	[[nodiscard]] static bool AsyncPresentAllowed() {
		static const bool allowed = std::getenv("KYTY_FRAMEGEN_SYNC_PRESENT") == nullptr;
		return allowed;
	}
	void PostAsync(Frame& frame, bool reuse) {
		const auto        begin = std::chrono::steady_clock::now();
		std::unique_lock lock(async_mutex);
		if (!async_thread.joinable()) {
			async_thread = std::thread([this] { AsyncLoop(); });
		}
		async_cv.wait(lock, [this] { return async_pending == nullptr; });
		const double blocked_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		async_post_count++;
		async_post_blocked_ms_sum += blocked_ms;
		async_post_blocked_ms_max = std::max(async_post_blocked_ms_max, blocked_ms);
		async_pending       = &frame;
		async_pending_reuse = reuse;
		async_cv.notify_all();
	}
	// Until the worker has presented everything handed to it (before a present or a blank frame on the vblank thread:
	// the swapchain and the present scheduler are used by one thread at a time).
	void DrainAsync() {
		std::unique_lock lock(async_mutex);
		async_cv.wait(lock, [this] { return async_pending == nullptr && !async_busy; });
	}
	[[nodiscard]] bool AsyncIdle() {
		std::lock_guard lock(async_mutex);
		return async_pending == nullptr && !async_busy;
	}
	void AsyncLoop() {
		LocalPlatform::SetThreadName("Kyty.Present");
		std::unique_lock lock(async_mutex);
		for (;;) {
			async_cv.wait(lock, [this] { return async_pending != nullptr || async_stop; });
			if (async_pending == nullptr) {
				return; // stopping
			}
			auto*      frame = async_pending;
			const bool reuse = async_pending_reuse;
			async_pending    = nullptr;
			async_busy       = true;
			async_cv.notify_all();
			lock.unlock();
			if (async_stop) {
				frames.Release(frame, reuse);
			} else {
				PresentNow(*frame, reuse);
			}
			lock.lock();
			async_busy = false;
			async_cv.notify_all();
		}
	}
	void StopAsync() {
		{
			std::lock_guard lock(async_mutex);
			async_stop = true;
			async_cv.notify_all();
		}
		if (async_thread.joinable()) {
			async_thread.join();
		}
		// A frame handed over after the worker's last look.
		if (async_pending != nullptr) {
			frames.Release(async_pending, async_pending_reuse);
			async_pending = nullptr;
		}
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		if (swapchain.Valid()) frames.SetFormat(swapchain.Format()); // desport: S-001 (no swapchain while 0x0)
	}

	Image& ResolveSurface(const ImageInfo& info) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		return image;
	}

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	std::atomic<uint64_t> presented_overlay_revision {0};
	// KYTY_FPS_HUD: the frame rate panel, its numbers refreshed once a second (the game's presented
	// frames times the frames DLSS-G shows for each).
	SystemOverlayHud                      hud;
	std::chrono::steady_clock::time_point hud_window {};
	uint32_t                              hud_frames = 0;

	// The present worker (PostAsync).
	std::mutex              async_mutex;
	std::condition_variable async_cv;
	std::thread             async_thread;
	Frame*                  async_pending       = nullptr;
	bool                    async_pending_reuse = false;
	bool                    async_busy          = false;
	bool                    async_stop          = false;
	uint32_t                async_post_count          = 0;   // under async_mutex
	double                  async_post_blocked_ms_sum = 0.0; // under async_mutex
	double                  async_post_blocked_ms_max = 0.0; // under async_mutex
	// Frame generation presents (PresentNow): the GPU wait before it and the Streamline present call.
	struct PresentTiming {
		uint32_t presents = 0;
		double   gpu_wait_sum = 0.0, gpu_wait_max = 0.0;
		double   call_sum = 0.0, call_max = 0.0;
		uint32_t over_vblank = 0; // gpu wait + call longer than a 60 Hz vblank
	} present_timing;
	void RecordPresentTiming(double gpu_wait_ms, double call_ms) {
		auto& t = present_timing;
		t.presents++;
		t.gpu_wait_sum += gpu_wait_ms;
		t.gpu_wait_max = std::max(t.gpu_wait_max, gpu_wait_ms);
		t.call_sum += call_ms;
		t.call_max = std::max(t.call_max, call_ms);
		t.over_vblank += gpu_wait_ms + call_ms > 1000.0 / 60.0 ? 1u : 0u;
		if (t.presents < 300) {
			return;
		}
		uint32_t posts = 0;
		double   blocked_sum = 0.0, blocked_max = 0.0;
		{
			std::lock_guard lock(async_mutex);
			posts       = async_post_count;
			blocked_sum = async_post_blocked_ms_sum;
			blocked_max = async_post_blocked_ms_max;
			async_post_count          = 0;
			async_post_blocked_ms_sum = 0.0;
			async_post_blocked_ms_max = 0.0;
		}
		std::printf("Frame generation present timing last %u presents: gpu_wait avg/max=%.2f/%.2f ms "
		            "present_call avg/max=%.2f/%.2f ms over_60hz_vblank=%u async_posts=%u vblank_blocked avg/max=%.2f/%.2f ms\n",
		            t.presents, t.gpu_wait_sum / t.presents, t.gpu_wait_max, t.call_sum / t.presents, t.call_max,
		            t.over_vblank, posts, posts != 0 ? blocked_sum / posts : 0.0, blocked_max);
		std::fflush(stdout);
		t = {};
	}

	// The shader prefetch's progress line while it runs, and for a few seconds after.
	std::chrono::steady_clock::time_point prefetch_finished {};
	void UpdateStatus(std::chrono::steady_clock::time_point now) {
		const auto progress = renderer.GetPipelineCache().GetPrefetchProgress();
		hud.status.clear();
		if (progress.total == 0) {
			return;
		}
		std::array<char, 128> line {};
		if (progress.done < progress.total) {
			std::snprintf(line.data(), line.size(), "Preparing shaders %zu%%  (%zu / %zu)", progress.done * 100 / progress.total,
			              progress.done, progress.total);
		} else {
			if (prefetch_finished == std::chrono::steady_clock::time_point {}) {
				prefetch_finished = now;
			}
			if (now - prefetch_finished >= std::chrono::seconds(3)) {
				return;
			}
			std::snprintf(line.data(), line.size(), "All shaders ready (%zu)", progress.total);
		}
		hud.status          = line.data();
		hud.status_fraction = static_cast<float>(progress.done) / static_cast<float>(progress.total);
	}

	[[nodiscard]] SystemOverlayHud* UpdateHud(bool reuse) {
		const bool enabled = FrameGen::FpsHudEnabled();
		const auto now = std::chrono::steady_clock::now();
		UpdateStatus(now);
		hud.notice = Loader::DemonsSoulsWarp::HudText();
		if (!enabled) {
			return hud.status.empty() && hud.notice.empty() ? nullptr : PlaceHud();
		}
		if (hud_window == std::chrono::steady_clock::time_point {}) {
			hud_window = now;
		}
		hud_frames += reuse ? 0u : 1u;
		const double seconds = std::chrono::duration<double>(now - hud_window).count();
		if (seconds >= 1.0) {
			const double   game  = hud_frames / seconds;
			const uint32_t shown = FrameGen::Enabled() ? FrameGen::PresentedFrames() : 1u;
			std::array<char, 32>  title {};
			std::array<char, 128> detail {};
			std::snprintf(title.data(), title.size(), "%.0f FPS", game * shown);
			if (shown > 1) {
				std::snprintf(detail.data(), detail.size(), "DLSS Frame Generation %ux  |  game %.0f fps",
				              shown, game);
			} else {
				std::snprintf(detail.data(), detail.size(), "game %.0f fps", game);
			}
			hud.title  = title.data();
			hud.detail = detail.data();
			hud_window = now;
			hud_frames = 0;
		}
		if (hud.title.empty() && hud.status.empty() && hud.notice.empty()) {
			return nullptr;
		}
		return PlaceHud();
	}
	SystemOverlayHud* PlaceHud() {
		hud.region = swapchain.PresentRegion();
		if (hud.region.extent.width == 0 || hud.region.extent.height == 0) {
			hud.region = vk::Rect2D {{0, 0}, swapchain.Extent()};
		}
		return &hud;
	}
#if defined(_WIN32)
	// Presents faster than the display refreshes only pile up in the compositor, and on Windows
	// a present blocked there holds the driver's device lock (see Swapchain::Present).
	std::chrono::steady_clock::time_point last_present {};
	std::chrono::nanoseconds              min_present_interval {-1}; // -1: not queried yet
	// A frame PresentDue turned away that no newer present replaced: the vblank thread shows it once the display can
	// take it (Presenter::PresentSkippedIfDue).
	bool skipped_unshown = false;

	// `commit`: the present goes ahead now (the next one is timed from it).
	[[nodiscard]] bool PresentDue(bool commit = true) {
		if (min_present_interval.count() < 0) {
			SDL_DisplayMode mode {};
			const int       display = window.window != nullptr ? SDL_GetWindowDisplayIndex(window.window) : -1;
			const int       refresh =
			    display >= 0 && SDL_GetCurrentDisplayMode(display, &mode) == 0 ? mode.refresh_rate : 0;
			min_present_interval = std::chrono::nanoseconds(refresh > 0 ? 1000000000LL / refresh : 0);
			LOGF("Presents limited to the display refresh: %d Hz\n", refresh);
		}
		const auto now = std::chrono::steady_clock::now();
		// A quarter of the interval as slack: a 60 Hz guest on a 60 Hz display flips at the same
		// rate, and with jitter every third present came a little early (40 fps shown).
		if (min_present_interval.count() > 0 && now - last_present < min_present_interval * 3 / 4) return false;
		if (commit) last_present = now;
		return true;
	}
#endif
};

// --hdr: an HDR10 (ST 2084 PQ) surface format the presentation blit can write. The swapchain uses it
// while the guest's frames are PQ; SDR frames (and before the first PQ frame) keep an SDR swapchain.
vk::SurfaceFormatKHR Swapchain::SelectFormat() {
	const auto& formats    = m_window.surface_capabilities.formats;
	auto&       graphics   = m_window.graphic_ctx;
	const auto  hdr_format = std::find_if(formats.begin(), formats.end(), [&](const vk::SurfaceFormatKHR& candidate) {
		return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT && IsHdrSwapchainFormat(candidate.format) &&
		       static_cast<bool>(graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
		                         vk::FormatFeatureFlagBits::eBlitDst);
	});
	const bool supported = Config::HdrEnabled() && hdr_format != formats.end();
	if (Config::HdrEnabled() && supported != m_hdr_supported.load(std::memory_order_relaxed)) {
		std::printf("HDR10 output: %s\n", supported ? "available" : "the display offers no HDR10 surface (SDR output)");
		std::fflush(stdout);
	}
	m_hdr_supported.store(supported, std::memory_order_relaxed);
	m_hdr_active = m_hdr_requested && supported;
	if (m_hdr_active) {
		return *hdr_format;
	}
	if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
		return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	}
	for (const auto& candidate: formats) {
		if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
			continue;
		}
		if (candidate.format == vk::Format::eB8G8R8A8Unorm || candidate.format == vk::Format::eR8G8B8A8Unorm) {
			return candidate;
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}

bool Swapchain::Create() { // desport: S-001
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	Common::LockGuard lock(m_window.mutex);
	EXIT_IF(graphics.screen_width == 0);
	EXIT_IF(graphics.screen_height == 0);
	const auto&       surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(graphics.screen_width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(graphics.screen_height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	// desport: S-001 never create a 0x0 swapchain (minimized window): leave it without one; the present path
	// skips presenting and tries again when the window is back.
	if (m_extent.width == 0 || m_extent.height == 0) {
		m_extent = {};
		LOGF("Swapchain: surface extent is 0x0, no swapchain for now\n");
		return false;
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	const auto format             = SelectFormat();
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped          = VK_TRUE;
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);
	if (Config::HdrEnabled()) {
		std::printf("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");
		std::fflush(stdout);
	}

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
	return true; // desport: S-001
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (graphics.present_queue != nullptr) {
		Common::LockGuard queue_lock(graphics.present_queue_mutex);
		RequireVulkanSuccess(graphics.present_queue.waitIdle(), "wait for present queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle      = nullptr;
	m_format      = vk::Format::eUndefined;
	m_extent      = {};
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		m_window.RunOnMainThread([this] { m_window.RecreateSurface(); });
#else
		m_window.RecreateSurface();
#endif
	}
	m_window.RefreshSurfaceCapabilities();
	(void)Create(); // desport: S-001 (false: 0x0 surface, Valid() says so)
}

Swapchain::Status Swapchain::AcquireNextImage() {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	m_image_index     = static_cast<uint32_t>(-1);
	// desport: S-001 a finite timeout (KYTY_ACQUIRE_TIMEOUT_MS, default 100, 0 = wait forever as before): a driver
	// that does not hand out images (minimized, covered window) costs the frame, not the whole emulation.
	static const uint64_t acquire_timeout_ns = [] {
		uint64_t ms = 100;
		if (const char* value = std::getenv("KYTY_ACQUIRE_TIMEOUT_MS"); value != nullptr && *value != '\0') {
			ms = std::strtoull(value, nullptr, 10);
		}
		return ms == 0 ? std::numeric_limits<uint64_t>::max() : ms * 1000000ull;
	}();
	static std::atomic_bool timing_out {false}; // desport: S-001 (log once per transition)
	const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
	    m_handle, acquire_timeout_ns, m_image_acquired[m_frame_index], nullptr, // desport: S-001
	    &m_image_index);
	if (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
		if (timing_out.exchange(false)) { // desport: S-001
			LOGF("DES_WINDOW acquire_recovered\n");
			std::printf("DES_WINDOW acquire_recovered\n");
			std::fflush(stdout);
		}
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eTimeout: // desport: S-001
		case vk::Result::eNotReady:
			if (!timing_out.exchange(true)) {
				LOGF("DES_WINDOW acquire_timeout (%llu ms)\n", static_cast<unsigned long long>(acquire_timeout_ns / 1000000ull));
				std::printf("DES_WINDOW acquire_timeout (%llu ms)\n", static_cast<unsigned long long>(acquire_timeout_ns / 1000000ull));
				std::fflush(stdout);
			}
			return Status::Timeout;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	EXIT_IF(m_image_index >= m_images.size());
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay(SystemOverlayHud* hud) {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount(), hud);
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, VulkanImage& source,
                                      bool draw_system_overlay) {
	if (source.state.layout != vk::ImageLayout::eTransferSrcOptimal) {
		EXIT("invalid prepared presentation image, vk_image=%p layout=%d\n",
		     static_cast<void*>(source.image), static_cast<int>(source.state.layout));
	}
	EXIT_IF(m_image_index >= m_images.size());
	auto vk_command = command.Handle();

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
	                           nullptr, 0, nullptr, 1, &to_transfer);

	// KYTY_PRESENT_ASPECT=fit: keep the frame's aspect ratio (black bars) instead of stretching
	// it over the window; a 3840x2160 frame on a 5120x2160 screen is then copied 1:1.
	static const bool aspect_fit = [] {
		const char* value = std::getenv("KYTY_PRESENT_ASPECT");
		return value != nullptr && std::string_view(value) == "fit";
	}();
	int dst_x = 0, dst_y = 0;
	int dst_w = static_cast<int>(m_extent.width), dst_h = static_cast<int>(m_extent.height);
	if (aspect_fit && source.extent.width != 0 && source.extent.height != 0) {
		const double scale = std::min(static_cast<double>(m_extent.width) / source.extent.width,
		                              static_cast<double>(m_extent.height) / source.extent.height);
		dst_w = std::max(1, static_cast<int>(std::lround(source.extent.width * scale)));
		dst_h = std::max(1, static_cast<int>(std::lround(source.extent.height * scale)));
		dst_x = (static_cast<int>(m_extent.width) - dst_w) / 2;
		dst_y = (static_cast<int>(m_extent.height) - dst_h) / 2;
		if (dst_w != static_cast<int>(m_extent.width) || dst_h != static_cast<int>(m_extent.height)) {
			const vk::ClearColorValue black {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
			vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, &black, 1,
			                           &to_transfer.subresourceRange);
			// (An image barrier: <windows.h> defines MemoryBarrier as a macro.)
			auto cleared          = to_transfer;
			cleared.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			cleared.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			cleared.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
			cleared.newLayout     = vk::ImageLayout::eTransferDstOptimal;
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
			                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &cleared);
		}
	}

	m_present_region = vk::Rect2D {{dst_x, dst_y}, {static_cast<uint32_t>(dst_w), static_cast<uint32_t>(dst_h)}};
	vk::ImageBlit region {};
	region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.srcSubresource.mipLevel       = 0;
	region.srcSubresource.baseArrayLayer = 0;
	region.srcSubresource.layerCount     = 1;
	region.srcOffsets[1].x               = static_cast<int>(source.extent.width);
	region.srcOffsets[1].y               = static_cast<int>(source.extent.height);
	region.srcOffsets[1].z               = 1;
	region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.dstSubresource.mipLevel       = 0;
	region.dstSubresource.baseArrayLayer = 0;
	region.dstSubresource.layerCount     = 1;
	region.dstOffsets[0].x               = dst_x;
	region.dstOffsets[0].y               = dst_y;
	region.dstOffsets[1].x               = dst_x + dst_w;
	region.dstOffsets[1].y               = dst_y + dst_h;
	region.dstOffsets[1].z               = 1;
	vk_command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal,
	                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1, &region,
	                     vk::Filter::eLinear);

	vk::ImageMemoryBarrier to_present {};
	to_present.sType         = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask = draw_system_overlay ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                     vk::AccessFlagBits::eColorAttachmentWrite
	                                               : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout     = draw_system_overlay ? vk::ImageLayout::eColorAttachmentOptimal
	                                               : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.image                           = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer,
	    draw_system_overlay ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                        : vk::PipelineStageFlagBits::eAllCommands,
	    vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1, &to_present);
	if (draw_system_overlay) {
		m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	return scheduler.Submit(submit);
}

Swapchain::Status Swapchain::Present() {
	EXIT_IF(m_image_index >= m_render_complete.size());
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	if (auto& graphics = m_window.graphic_ctx; graphics.present_queue != nullptr) {
		// Its own queue: a present blocked in the kernel does not hold up other submissions.
		Common::LockGuard lock(graphics.present_queue_mutex);
		result = graphics.present_queue.presentKHR(&present);
	} else {
		Common::LockGuard lock(graphics.queue_mutex);
		result = graphics.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto*             frame = m_impl->frames.Acquire();
	frame->hdr              = hdr;
	frame->blank            = false;
	RenderLockGuard render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}

	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	frame->fg.valid         = false;
	frame->fg.token_index   = 0;     // desport: phase B (PrepareInputs sets it with DLSS SR + DLSS-G)
	frame->fg.constants_set = false; // desport
	if (FrameGen::CaptureRequested()) {
		FrameGen::PrepareInputs(m_impl->window.graphic_ctx, m_impl->renderer.GetTextureCache(),
		                        buffer, frame->fg);
	} else if (FrameGen::SrActive()) { // desport: DLSS SR without DLSS-G: the frame's camera ends at the flip
		FrameGen::EndSrFrame();         // desport: (PrepareInputs does it with DLSS-G)
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire();
	// Into the GPU thread's command buffer (its flip) under the renderer's lock; the present thread's blank frame
	// (no producer) touches only the presenter's own scheduler and frame, as Present does.
	std::optional<RenderLockGuard> render_lock;
	if (producer != nullptr) render_lock.emplace(m_impl->renderer.GetMutex());
	frame->fg.valid = false;
	frame->fg.token_index   = 0;     // desport: phase B, a blank frame takes a new Streamline frame index at present
	frame->fg.constants_set = false; // desport
	frame->hdr      = false;
	frame->blank    = true;
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		m_impl->DrainAsync(); // the present worker records on the same scheduler
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

Presenter::Frame* Presenter::PrepareLastFrame() {
	return m_impl->frames.AcquireLast();
}

bool Presenter::PresentSkippedIfDue() {
#if defined(_WIN32)
	// A game above the display's rate had every frame that came within a refresh of the one before turned away, and
	// the refresh after it showed nothing new when the next frame was late (a 69 fps game on a 60 Hz display: 52
	// frames a second shown). Each refresh shows the latest frame now. (Frame generation paces its own frames.)
	// (The present worker owns the presenting state while it runs: frame generation was just switched off.)
	if (FrameGen::Enabled() || !m_impl->AsyncIdle()) return false;
	if (!m_impl->skipped_unshown || !m_impl->PresentDue(false)) return false;
	auto* frame = PrepareLastFrame();
	if (frame == nullptr) {
		m_impl->skipped_unshown = false; // (the GPU thread took it for a newer frame)
		return false;
	}
	Present(*frame, true);
	return true;
#else
	return false;
#endif
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::IsHdrSupported() const noexcept {
	return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

bool Presenter::FlipRateApplies(const Frame& frame) noexcept {
	return !FrameGen::Enabled() || frame.fg.valid;
}

void Presenter::Present(Frame& frame, bool reuse) {
	KYTY_PROFILER_FUNCTION();
	m_impl->frames.ValidateForPresent(&frame, reuse);
	// desport: S-001 minimized window: retire the flip without presenting (and without waiting for the FG worker)
	if (m_impl->window.minimized.load(std::memory_order_acquire)) {
		m_impl->frames.Release(&frame, true);
		return;
	}
	if (FrameGen::Enabled() && Impl::AsyncPresentAllowed()) {
		m_impl->PostAsync(frame, reuse);
		return;
	}
	m_impl->DrainAsync();
	m_impl->PresentNow(frame, reuse);
}

void Presenter::Impl::PresentNow(Frame& frame, bool reuse) {
	KYTY_PROFILER_FUNCTION();
	// desport: S-001 minimized window (also for a frame the FG worker already holds): nothing to show and the driver may
	// block acquire/present; the frame goes back to the pool and the flip retires as usual (no FrameGen::BeforePresent).
	if (window.minimized.load(std::memory_order_acquire)) {
		frames.Release(&frame, true);
		return;
	}
	// desport: S-001 window back: refresh the surface capabilities and recreate the swapchain before the first acquire;
	// also when the last recreate found a 0x0 surface and left no swapchain.
	if (window.surface_dirty.exchange(false, std::memory_order_acq_rel) || !swapchain.Valid()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
		if (!swapchain.Valid()) {
			frames.Release(&frame, true);
			return;
		}
	}
#if defined(_WIN32)
	if (!PresentDue()) {
		// A frame the display could not show anyway (menus and movies run far above it); it
		// stays the latest frame for an idle refresh, and the next vblank that finds the display
		// ready and no newer frame shows it (PresentSkippedIfDue).
		skipped_unshown = true;
		frames.Release(&frame, true);
		return;
	}
#endif
#if defined(KYTY_LOCAL_VULKAN_RECORDING) && defined(_WIN32)
	// The frame's rendering may still sit in the recording worker's queue (KYTY_DEFERRED_SUBMIT).
	// On Windows, presenting before it reaches the driver blocks vkQueuePresentKHR in the
	// kernel while it holds the queue lock that submission needs: a deadlock. Wait (no locks
	// held) until it has been submitted; this does not wait for the GPU.
	LocalVulkanRecording::WaitDeferredSubmits();
#endif

	const auto overlay_visual = GetSystemOverlayVisualState();
	auto*      overlay_hud    = UpdateHud(reuse);
	// The guest's own frames choose the color space (PQ or SDR); black and blank flips keep the current one.
	if (!frame.blank && swapchain.SetHdr(frame.hdr)) {
		RecoverSwapchain(Swapchain::Status::Recreate);
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		if (!swapchain.Valid()) break; // desport: S-001 recreate found a 0x0 surface
		auto status = swapchain.AcquireNextImage();
		if (status == Swapchain::Status::Timeout) { // desport: S-001 drop the frame, do not block the vblank thread
			frames.Release(&frame, true);
			return;
		}
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}
		{
			// The present thread records and submits on its own scheduler (present_scheduler: its own pool, timeline
			// and command buffers; the queue has its own lock), from the presenter's frame image into the swapchain's:
			// nothing the GPU thread uses. Under the renderer's lock, its GPU thread waited for the whole recording
			// and submission once a frame (~60 us, ~110 us with the wake-up). Frame generation shares the renderer's.
			std::optional<RenderLockGuard> render_lock;
			if (FrameGen::Enabled()) render_lock.emplace(renderer.GetMutex());
			auto&      command             = present_scheduler.BeginCommand();
			const bool draw_system_overlay =
			    (overlay_visual.active || overlay_hud != nullptr) && swapchain.PrepareSystemOverlay(overlay_hud);
			if (overlay_hud != nullptr && draw_system_overlay && FrameGen::Enabled()) {
				// The panel's motion vectors are zeroed (a margin around it).
				const auto& region = overlay_hud->region;
				const auto& drawn  = overlay_hud->drawn;
				const float w = static_cast<float>(region.extent.width), h = static_cast<float>(region.extent.height);
				const float margin = 8.0f;
				FrameGen::SetStaticRect(
				    (static_cast<float>(drawn.offset.x - region.offset.x) - margin) / w,
				    (static_cast<float>(drawn.offset.y - region.offset.y) - margin) / h,
				    (static_cast<float>(drawn.offset.x - region.offset.x) + static_cast<float>(drawn.extent.width) + margin) / w,
				    (static_cast<float>(drawn.offset.y - region.offset.y) + static_cast<float>(drawn.extent.height) + margin) / h);
			}
			swapchain.RecordPresentCommands(command, frame.image, draw_system_overlay);
			frame.present_tick = swapchain.Submit(present_scheduler);
		}
		const auto wait_begin = std::chrono::steady_clock::now();
#if defined(_WIN32)
		// The Windows driver waits in vkQueuePresentKHR, inside the kernel, for the semaphore's
		// GPU work while it holds a device-wide lock: every queue submission waits behind it,
		// including those the GPU work may still depend on (a deadlock). Wait for the GPU here,
		// holding nothing; the present on its own queue then finds the semaphore signaled.
		present_scheduler.Wait(frame.present_tick);
#endif
		const auto call_begin = std::chrono::steady_clock::now();
		// KYTY_FRAMEGEN: Streamline takes the present (vkQueuePresentKHR is its hook) and adds
		// the generated frames; an idle refresh of the last frame carries no new inputs.
		const bool streamline_present = FrameGen::PresentIntegrationReady();
		if (streamline_present) {
			FrameGen::BeforePresent(frame.fg.valid && !frame.reusing_last ? &frame.fg : nullptr,
			                        {frame.image.extent.width, frame.image.extent.height}, swapchain.Extent(),
			                        swapchain.PresentRegion(), swapchain.Format(), swapchain.ImageCount(),
			                        frame.reusing_last,
			                        // desport: phase B, the frame index the DLSS SR used (a reused frame takes a new one)
			                        frame.reusing_last ? 0u : frame.fg.token_index,
			                        !frame.reusing_last && frame.fg.constants_set);
		}
		status = swapchain.Present();
		if (streamline_present) {
			FrameGen::AfterPresent();
		}
		if (FrameGen::Enabled()) {
			const auto call_end = std::chrono::steady_clock::now();
			RecordPresentTiming(std::chrono::duration<double, std::milli>(call_begin - wait_begin).count(),
			                    std::chrono::duration<double, std::milli>(call_end - call_begin).count());
		}
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
#if defined(_WIN32)
		skipped_unshown = false;
#endif
		window.UpdateTitle();
		frames.Release(&frame, true);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	frames.Release(&frame, reuse);
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
