#pragma once
// Local diagnostic: a GPU timestamp after each recorded draw, dispatch or native XPR run while
// a live `tracem` runs (live-trace.h). The timestamp is recorded in order with the command, so
// the recording worker replays it right after; it costs one relaxed load otherwise.

#include "live-trace.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace LiveTrace {

// MarkVulkan: an emulator-side command (copy, fill, clear, blit, resolve, barrier or a dispatch of
// its own), tagged with a VulkanMark id (generate-vulkan-recording.py).
enum MarkKind : uint64_t { MarkDraw = 1, MarkDispatch = 2, MarkNativeXpr = 3, MarkVulkan = 4 };

// Set on the GPU (render) thread: only its command buffers belong to ticks.
inline thread_local bool g_mark_thread = false;
// The scheduler's command buffer being recorded (CommandScheduler::BeginCommand): other command
// buffers the thread records (presentation, one-off transfers) get no marks.
inline thread_local VkCommandBuffer g_mark_command = VK_NULL_HANDLE;
// Nonzero while a guest draw or dispatch is recorded: its own dispatch carries the guest mark.
inline thread_local uint32_t g_mark_scope = 0;

inline void GpuMarkNow(vk::CommandBuffer command, MarkKind kind, uint64_t shader) {
	if (!g_marks_on.load(std::memory_order_relaxed) || g_mark_pool == nullptr || !command) return;
	const auto slot = g_mark_next.fetch_add(1, std::memory_order_relaxed);
	if (slot >= MarkSlots) return;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteTimestamp(static_cast<VkCommandBuffer>(command),
	                                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                                                  static_cast<VkQueryPool>(g_mark_pool), slot);
	Event(GpuMark, slot, (shader & 0x0fffffffffffffffull) | kind << 60u);
}

// Marks the command recorded while this object lives; `handle` yields the command buffer then.
template <typename Handle>
struct MarkAfter {
	Handle   handle;
	MarkKind kind;
	uint64_t shader;
	MarkAfter(Handle h, MarkKind k, uint64_t s): handle(h), kind(k), shader(s) { ++g_mark_scope; }
	MarkAfter(const MarkAfter&)            = delete;
	MarkAfter& operator=(const MarkAfter&) = delete;
	~MarkAfter() {
		--g_mark_scope;
		if (g_marks_on.load(std::memory_order_relaxed)) GpuMarkNow(handle(), kind, shader);
	}
};
template <typename Handle>
MarkAfter(Handle, MarkKind, uint64_t) -> MarkAfter<Handle>;

// generate-vulkan-recording.py: after an emulator-side work command on the GPU thread.
inline void VulkanMark(VkCommandBuffer command, uint32_t id, bool dispatch) {
	if (!g_marks_on.load(std::memory_order_relaxed) || !g_mark_thread || command != g_mark_command ||
	    (dispatch && g_mark_scope != 0))
		return;
	GpuMarkNow(vk::CommandBuffer(command), MarkVulkan, id);
}

} // namespace LiveTrace
