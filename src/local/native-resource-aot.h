#ifndef KYTY_LINEAR_SRT_AOT_H
#define KYTY_LINEAR_SRT_AOT_H

#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#define KYTY_LOCAL_SRT_AOT 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) && defined(__x86_64__)
#define KYTY_LOCAL_SRT_AOT 1
#include <dlfcn.h>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {
#if defined(KYTY_LOCAL_SRT_AOT)
// The compiled library: a DLL on Windows (tools/local/compile-srt-aot-windows.py), a shared
// object on Linux (tools/local/compile-srt-aot.py).
#if defined(_WIN32)
inline void* OpenSrtAotLibrary(const char* path) { return reinterpret_cast<void*>(LoadLibraryA(path)); }
inline void* SrtAotSymbol(void* module, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
}
#else
inline void* OpenSrtAotLibrary(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* SrtAotSymbol(void* module, const char* name) { return dlsym(module, name); }
#endif
inline std::string EmitLinearSrtAot(const LinearSrtPlan& plan, bool packed = false) {
    using Kind = LinearSrtPlan::Kind;
    std::ostringstream out;
    const auto v = [](uint32_t i) { return "v" + std::to_string(i); };
    std::vector<std::string> stores(plan.nodes.size());
    if (packed) {
        size_t cursor = 0;
        for (uint32_t d = 0; d < plan.descriptor_sizes.size(); ++d) {
            for (uint32_t word = 0; word < plan.descriptor_sizes[d]; ++word) {
                const auto node = plan.descriptor_words[cursor++];
                if (node == UINT32_MAX) continue; // caller initializes inactive outputs
                stores[node] += "KytySrtAot::StoreDescriptor(output," + std::to_string(d) + ',' +
                                std::to_string(word) + ',' + v(node) + ");\n";
            }
        }
        for (uint32_t i = 0; i < plan.flat_words.size(); ++i) {
            const auto node = plan.flat_words[i];
            if (node != UINT32_MAX)
                stores[node] += "output->flat_data[" + std::to_string(i) + "] = uint32_t(" + v(node) + ");\n";
        }
    } else {
        for (uint32_t i = 0; i < plan.nodes.size(); ++i)
            stores[i] = "values[" + std::to_string(i) + "] = " + v(i) + ";\n";
    }
    for (uint32_t i = 0; i < plan.nodes.size(); ++i) out << "uint64_t " << v(i) << " = 0;\n";
    const auto emit = [&](uint32_t i) {
        const auto& n = plan.nodes[i];
        const auto a = v(n.args[0]), b = v(n.args[1]);
        std::string expression;
        switch (n.kind) {
        case Kind::Immediate: expression = std::to_string(n.immediate) + "ull"; break;
        case Kind::User:
            out << "if (r->user_count <= " << n.immediate << "ull) return false;\n";
            expression = "r->user_data[" + std::to_string(n.immediate) + "ull]"; break;
        case Kind::Base: expression = "r->shader_base"; break;
        case Kind::Read:
            out << "if (!KytySrtAot::Read<" << (n.op == ValueOpcode::ReadConstBuffer) << ',' << n.clean
                << ',' << int64_t(int32_t(n.immediate)) << ">(r," << a << ',' << b << ',' << v(n.args[2])
                << ',' << (n.op == ValueOpcode::ReadConstBuffer ? v(n.args[3]) : "0") << ',' << v(i)
                << ")) return false;\n";
            break;
        case Kind::Unary:
            if (n.op == ValueOpcode::CompositeExtractU64)
                expression = "uint32_t(" + a + (n.immediate ? " >> 32)" : ")");
            else if (n.op == ValueOpcode::BitwiseNot32) expression = "uint32_t(~uint32_t(" + a + "))";
            else expression = "!(" + a + ")";
            break;
        case Kind::Select: expression = a + " ? " + b + " : " + v(n.args[2]); break;
        case Kind::Binary: {
            const auto a32 = "uint32_t(" + a + ")", b32 = "uint32_t(" + b + ")";
            switch (n.op) {
            case ValueOpcode::IAdd32: expression = "uint32_t(" + a32 + '+' + b32 + ')'; break;
            case ValueOpcode::ISub32: expression = "uint32_t(" + a32 + '-' + b32 + ')'; break;
            case ValueOpcode::IMul32: expression = "uint32_t(" + a32 + '*' + b32 + ')'; break;
            case ValueOpcode::IAdd64: expression = a + '+' + b; break;
            case ValueOpcode::ISub64: expression = a + '-' + b; break;
            case ValueOpcode::IMul64: expression = a + '*' + b; break;
            case ValueOpcode::BitwiseAnd32: expression = a32 + '&' + b32; break;
            case ValueOpcode::BitwiseAnd64: expression = a + '&' + b; break;
            case ValueOpcode::BitwiseOr32: expression = a32 + '|' + b32; break;
            case ValueOpcode::BitwiseXor32: expression = a32 + '^' + b32; break;
            case ValueOpcode::ShiftLeftLogical32: expression = "uint32_t(" + a32 + " << (" + b + " & 31))"; break;
            case ValueOpcode::ShiftLeftLogical64: expression = a + " << (" + b + " & 63)"; break;
            case ValueOpcode::ShiftRightLogical32: expression = a32 + " >> (" + b + " & 31)"; break;
            case ValueOpcode::ShiftRightLogical64: expression = a + " >> (" + b + " & 63)"; break;
            case ValueOpcode::ShiftRightArithmetic32: expression = "uint32_t(int32_t(" + a + ") >> (" + b + " & 31))"; break;
            case ValueOpcode::ShiftRightArithmetic64: expression = "uint64_t(int64_t(" + a + ") >> (" + b + " & 63))"; break;
            case ValueOpcode::CompositeConstructU64: expression = "uint64_t(" + a32 + ") | (" + b + " << 32)"; break;
            case ValueOpcode::IEqual32: expression = a32 + " == " + b32; break;
            case ValueOpcode::INotEqual32: expression = a32 + " != " + b32; break;
            case ValueOpcode::ULessThan32: expression = a32 + " < " + b32; break;
            case ValueOpcode::UGreaterThan32: expression = a32 + " > " + b32; break;
            case ValueOpcode::UGreaterThanEqual32: expression = a32 + " >= " + b32; break;
            case ValueOpcode::UMin32: expression = a32 + " < " + b32 + " ? " + a32 + " : " + b32; break;
            case ValueOpcode::LogicalAnd: expression = "bool(" + a + ") && bool(" + b + ')'; break;
            case ValueOpcode::LogicalOr: expression = "bool(" + a + ") || bool(" + b + ')'; break;
            case ValueOpcode::LogicalXor: expression = "bool(" + a + ") != bool(" + b + ')'; break;
            default: std::abort();
            }
            break;
        }
        }
        if (!expression.empty()) out << v(i) << " = " << expression << ";\n";
        out << stores[i];
    };
    for (uint32_t i = 0, next_group = 0; i < plan.nodes.size(); ++i) {
        if (next_group == plan.read_groups.size() || plan.read_groups[next_group].indices[0] != i) {
            emit(i); continue;
        }
        const auto& group = plan.read_groups[next_group++];
        const uint32_t last = group.indices[group.count - 1];
        for (uint32_t j = i + 1; j < last; ++j) if (plan.nodes[j].kind == Kind::Immediate) emit(j);
        const auto& first = plan.nodes[i];
        const auto relative = [&](const auto& node) {
            const int64_t offset = uint32_t(plan.nodes[node.args[2]].immediate);
            const int64_t imm = int32_t(node.immediate);
            return node.op == ValueOpcode::ReadConstBuffer ? (imm + offset) & ~int64_t{3}
                                                          : (imm & ~int64_t{3}) + (offset & ~int64_t{3});
        };
        out << "{ uint32_t words[" << group.count << "];\nif (KytySrtAot::ReadSpan<"
            << (first.op == ValueOpcode::ReadConstBuffer) << ',' << first.clean << ',' << relative(first)
            << ',' << relative(plan.nodes[last]) << ',' << group.count << ">(r," << v(first.args[0])
            << ',' << v(first.args[1]) << ',' << (first.op == ValueOpcode::ReadConstBuffer ? v(first.args[3]) : "0")
            << ",words)) {\n";
        for (uint32_t j = 0; j < group.count; ++j)
            out << v(group.indices[j]) << " = words[" << j << "];\n" << stores[group.indices[j]];
        out << "} else {\n";
        for (uint32_t j = i; j <= last; ++j) emit(j);
        out << "} }\n";
        i = last;
    }
    out << "return true;\n";
    return out.str();
}

inline void PrepareLinearSrtAot(LinearSrtPlan& plan) {
    const auto* directory = std::getenv("KYTY_SRT_AOT_EXPORT");
    const auto* library = std::getenv("KYTY_SRT_AOT_LIBRARY");
    if ((!directory || !*directory) && (!library || !*library)) return;
    // Compile-time only; no hash, file access or dynamic linking per draw.
    const auto body = EmitLinearSrtAot(plan);
    const auto packed = EmitLinearSrtAot(plan, true);
    const auto signature = "kyty-srt-aot:" + std::to_string(KytySrtAot::AbiVersion) + '\n' + body + "packed:\n" + packed;
    uint64_t hash = 14695981039346656037ull;
    for (const auto byte : signature) hash = (hash ^ uint8_t(byte)) * 1099511628211ull;
    const auto symbol = fmt::format("kyty_srt_aot_{:016x}", hash);
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    if (directory && *directory) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (!error) {
            const auto path = std::filesystem::path(directory) / (symbol + ".cpp");
            if (!std::filesystem::exists(path)) {
                std::ofstream file(path);
                file << "#include \"SrtAotAbi.h\"\n"
                     << "static_assert(KytySrtAot::AbiVersion == " << KytySrtAot::AbiVersion << ");\n"
                     << "extern \"C\" KYTY_SRT_AOT_EXPORT const char* " << symbol
                     << "_signature() { return R\"KYTY(" << signature << ")KYTY\"; }\n"
                     << "extern \"C\" KYTY_SRT_AOT_EXPORT bool " << symbol
                     << "(KytySrtAot::Runtime* r, uint64_t* values) {\n" << body << "}\n"
                     << "extern \"C\" KYTY_SRT_AOT_EXPORT bool " << symbol
                     << "_materialize(KytySrtAot::Runtime* r, const KytySrtAot::Outputs* output) {\n" << packed << "}\n";
            }
        }
    }
    if (!library || !*library) return;
    // One explicitly selected, immutable local library lives for the process.
    // Full source-signature equality makes hash collisions harmless misses.
    static void* module = OpenSrtAotLibrary(library);
    if (!module) return;
    const auto get_signature = reinterpret_cast<const char* (*)()>(SrtAotSymbol(module, (symbol + "_signature").c_str()));
    const auto function = reinterpret_cast<KytySrtAot::Function>(SrtAotSymbol(module, symbol.c_str()));
    const auto materialize = reinterpret_cast<KytySrtAot::MaterializeFunction>(SrtAotSymbol(module, (symbol + "_materialize").c_str()));
    if (!get_signature || !function || !materialize || signature != get_signature()) return;
    plan.aot_function = function;
    plan.aot_materialize = materialize;
}
#else
inline void PrepareLinearSrtAot(LinearSrtPlan&) {}
#endif
} // namespace Libs::Graphics::ShaderRecompiler::IR
#endif
