#pragma once
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include "native-preparation-scratch.h"
#include "native-resource-state.h"

#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace LocalNativeResource {

struct OutputWorkspace {
    std::vector<DescriptorValue> evaluated;
    std::vector<uint32_t> flattened;
};

// With kyty_local_srt_native_mode, the ahead-of-time compiled plan writes the
// final descriptors and flattened SRT directly. Empty when the switch is off
// or the plan has no compiled function; otherwise whether evaluation succeeded.
inline std::optional<bool> Try(const LinearSrtPlan& plan, const SrtRuntime& runtime,
                               std::vector<DescriptorValue>& results, std::vector<uint32_t>& flat,
                               std::vector<uint8_t>& active_sources) {
    if (kyty_local_srt_native_mode.load(std::memory_order_relaxed) == 0 || !plan.aot_materialize) return {};
    KytySrtAot::Runtime abi{runtime.user_data.data(), runtime.user_data.size(), runtime.shader_base,
        runtime.userdata, runtime.read_memory, runtime.read_specialization_memory,
        runtime.try_read_memory_span, true};
    NativePreparationScratch<OutputWorkspace> storage;
    auto& evaluated = storage.Get().evaluated;
    auto& flattened = storage.Get().flattened;
    evaluated.assign(plan.descriptor_sizes.size(), {});
    for (size_t i = 0; i < evaluated.size(); ++i) evaluated[i].dword_count = plan.descriptor_sizes[i];
    flattened.assign(plan.flat_words.size(), 0u);
    const KytySrtAot::Outputs output{reinterpret_cast<unsigned char*>(evaluated.data()),
                                   sizeof(DescriptorValue), flattened.data()};
    if (!plan.aot_materialize(&abi, &output)) return false;
    results.swap(evaluated);
    flat.swap(flattened);
    if (plan.active_sources.empty()) active_sources.assign(plan.active_count, 1u);
    else active_sources = plan.active_sources;
    return true;
}
} // namespace LocalNativeResource
} // namespace Libs::Graphics::ShaderRecompiler::IR
