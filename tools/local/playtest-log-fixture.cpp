// Standalone evidence/overhead fixture; not linked into the emulator.
#include "kyty-render-cost.h"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    rlimit core {0, 0}; setrlimit(RLIMIT_CORE, &core);
    LocalPlaytest::Initialize();
    const auto now = LocalRenderCost::Now();
    if (argc > 1 && !std::strcmp(argv[1], "micro")) {
        for (unsigned i = 0; i < 1'000'000; ++i) {
            LocalRenderCost::SlowOperation wait("GpuWait", i);
        }
        std::printf("million_wait_scopes_ns=%llu\n", (unsigned long long)(LocalRenderCost::Now() - now));
        return 0;
    }
    if (argc > 1 && !std::strcmp(argv[1], "crash")) {
        LocalPlaytest::Begin("ShaderCompile", 0xabc, 4096, now);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        LocalPlaytest::Fault(0x900ac91ff, 0x123456, 0xdeadbeef, 11);
        std::raise(SIGABRT);
    }
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 8; ++i) workers.emplace_back([i] {
        for (unsigned j = 0; j < 50; ++j) {
            const auto start = LocalRenderCost::Now();
            const auto operation = LocalPlaytest::Begin("ShaderCompile", i * 50 + j, 4096, start);
            LocalPlaytest::End("ShaderCompile", operation, i * 50 + j, 4096, start, start + 6'000'000, "file path\nwith%=fields", 0, 0);
        }
    });
    for (auto& worker : workers) worker.join();
    LocalPlaytest::Frame(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    LocalPlaytest::Frame(2);
    // A long wait without a completion event exercises the external watchdog.
    const auto pending = LocalPlaytest::Begin("GraphicsPipeline", 0xabc, 0, LocalRenderCost::Now());
    std::this_thread::sleep_for(std::chrono::milliseconds(4500));
    LocalPlaytest::End("GraphicsPipeline", pending, 0xabc, 0, now, LocalRenderCost::Now(), "driver", 0, 0);
    LocalPlaytest::Frame(3);
    return 0;
}
