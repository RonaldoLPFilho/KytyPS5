#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics): m_graphics(graphics) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	// desport: scene_bias_256 (KYTY_TEX_DLSS_BIAS): an extra LOD bias in 1/256 units for a pixel shader's material
	// sampler (see samplerCache.cpp); a separate sampler per value.
	vk::Sampler GetSampler(const ShaderSamplerResource& r, int32_t scene_bias_256 = 0);

	// desport: KYTY_TEX_DLSS_BIAS is set (auto or a value): the callers pass scene_bias_256.
	[[nodiscard]] static bool SceneBiasEnabled();
	// desport: the scene bias to pass (1/256 units), given the DLSS render/output LOD offset FrameGen measured.
	[[nodiscard]] static int32_t SceneBias256(int32_t dlss_bias_256);

private:
	using SamplerKey = std::array<uint32_t, 5>; // desport: + the scene bias (KYTY_TEX_DLSS_BIAS)

	struct SamplerKeyHash {
		std::size_t operator()(const SamplerKey& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	GraphicContext&                                             m_graphics;
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
