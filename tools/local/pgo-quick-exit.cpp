// Link this object into an instrumented Clang build only. The emulator uses
// quick_exit while guest threads are live, which skips LLVM's atexit writer.
// Keeping the hook in a local object avoids a profiling dependency in the core.
#include <cstdlib>

extern "C" int __llvm_profile_write_file();

namespace {
const int registered = std::at_quick_exit([] { (void)__llvm_profile_write_file(); });
}
