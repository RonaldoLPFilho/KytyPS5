#ifndef KYTY_SRT_AOT_ABI_H
#define KYTY_SRT_AOT_ABI_H

#include <cstdint>
#include <cstring>

// Exported entry points of a compiled plan library.
#if defined(_WIN32)
#define KYTY_SRT_AOT_EXPORT __declspec(dllexport)
#else
#define KYTY_SRT_AOT_EXPORT __attribute__((visibility("default")))
#endif

// Standalone interface for locally compiled resource plans. No STL layouts,
// emulator pointers embedded in code, or guest memory contents enter the file.
namespace KytySrtAot {
constexpr uint32_t AbiVersion = 2;
using Reader = bool (*)(void*, uint64_t, uint32_t*);
using SpanReader = bool (*)(void*, uint64_t, uint32_t*, uint32_t, bool);
struct Runtime {
    const uint32_t* user_data;
    uint64_t user_count;
    uint64_t shader_base;
    void* userdata;
    Reader raw_reader;
    Reader clean_reader;
    SpanReader span_reader;
    bool spans;
    uint64_t groups = 0, words = 0;
};
using Function = bool (*)(Runtime*, uint64_t*);
struct Outputs {
    // Byte stride is supplied by C++, avoiding any assumed DescriptorValue ABI.
    unsigned char* descriptor_data;
    uint64_t descriptor_stride;
    uint32_t* flat_data;
};
using MaterializeFunction = bool (*)(Runtime*, const Outputs*);
inline void StoreDescriptor(const Outputs* output, uint32_t descriptor, uint32_t word, uint64_t value) {
    const auto narrowed = uint32_t(value);
    std::memcpy(output->descriptor_data + descriptor * output->descriptor_stride + word * 4, &narrowed, 4);
}

template<bool Buffer, int32_t Immediate>
inline bool Address(uint64_t low, uint64_t high, uint64_t offset, uint64_t records,
                    uint64_t& address) {
    constexpr uint64_t mask = 0x0000ffffffffffffull;
    const uint64_t base = ((high << 32) | uint32_t(low)) & mask;
    if constexpr (Buffer) {
        if constexpr (Immediate < 0) return false;
        const auto bytes = uint64_t(int64_t(Immediate)) + uint32_t(offset);
        const auto aligned = bytes & ~uint64_t{3};
        const auto stride = (uint32_t(high) >> 16) & 0x3fff;
        const uint64_t size = stride == 0 ? uint32_t(records) : uint64_t(stride) * uint32_t(records);
        if (aligned > size || size - aligned < 4) return false;
        address = ((base & ~uint64_t{3}) + bytes) & ~uint64_t{3};
    } else {
        const int64_t relative = (int64_t(Immediate) & ~int64_t{3}) + int64_t(uint32_t(offset) & ~3u);
        const auto aligned = base & ~uint64_t{3};
        if (relative < 0) {
            if (uint64_t(-relative) > aligned) return false;
            address = aligned - uint64_t(-relative);
        } else {
            if (uint64_t(relative) > mask - aligned) return false;
            address = aligned + uint64_t(relative);
        }
    }
    return true;
}

template<bool Buffer, bool Clean, int32_t Immediate>
inline bool Read(Runtime* runtime, uint64_t low, uint64_t high, uint64_t offset,
                 uint64_t records, uint64_t& output) {
    uint64_t address = 0;
    if (!Address<Buffer, Immediate>(low, high, offset, records, address)) return false;
    uint32_t word = 0;
    auto reader = Clean ? runtime->clean_reader : runtime->raw_reader;
    if (reader) {
        if (!reader(runtime->userdata, address, &word)) return false;
    } else {
        if constexpr (Clean) return false;
        std::memcpy(&word, reinterpret_cast<const void*>(address), 4);
    }
    output = word;
    return true;
}

template<bool Buffer, bool Clean, int64_t Start, int64_t End, uint32_t Count>
inline bool ReadSpan(Runtime* runtime, uint64_t low, uint64_t high, uint64_t records,
                     uint32_t* words) {
    static_assert(Count >= 2 && Count <= 16 && End - Start == (Count - 1) * 4);
    if (!runtime->spans || !runtime->span_reader) return false;
    constexpr uint64_t mask = 0x0000ffffffffffffull;
    const uint64_t base = ((high << 32) | uint32_t(low)) & (mask & ~uint64_t{3});
    if constexpr (Buffer) {
        static_assert(Start >= 0);
        const auto stride = (uint32_t(high) >> 16) & 0x3fff;
        const uint64_t size = stride == 0 ? uint32_t(records) : uint64_t(stride) * uint32_t(records);
        if (size < uint64_t(End) + 4) return false;
    } else {
        if constexpr (Start < 0) if (base < uint64_t(-Start)) return false;
        if constexpr (End >= 0) if (base > mask - uint64_t(End)) return false;
    }
    if (!runtime->span_reader(runtime->userdata, base + uint64_t(Start), words, Count, Clean)) return false;
    ++runtime->groups;
    runtime->words += Count;
    return true;
}
} // namespace KytySrtAot
#endif
