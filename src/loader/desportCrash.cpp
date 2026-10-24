// SPDX-License-Identifier: GPL-2.0-only
// desport: crash record. A real crash (an unhandled host exception, an EXIT/abort of the engine, a Vulkan device loss
// that ends in EXIT) writes one minidump and one "DES_CRASH" line, so every soak crash comes with evidence.
//
// Not touched: the vectored exception handler's normal work (GPU-protected page faults, red zone, guarded copies).
// Those faults are resolved before anything here runs; only what the engine itself ends with EXIT, and what nobody
// handles at all (SetUnhandledExceptionFilter), reaches this file.
#include "loader/desport.h"

#include "common/assert.h"
#include "local-platform.h"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fmt/format.h>
#include <string>

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// (After windows.h.)
#include <dbghelp.h>
#include <intrin.h>

namespace Desport::Crash {

namespace {

// The faulting context of the host exception the engine is about to end with EXIT (runtimeLinker.cpp), per thread.
struct NotedException {
	bool    valid = false;
	CONTEXT context {};
	DWORD   code = 0;
	DWORD64 pc   = 0;
};
thread_local NotedException t_noted;

// One crash record per process: a second crash while the first is written (or after it) only returns.
std::atomic_bool g_recorded {false};

using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE, PMINIDUMP_EXCEPTION_INFORMATION,
                                          PMINIDUMP_USER_STREAM_INFORMATION, PMINIDUMP_CALLBACK_INFORMATION);

std::string OneLine(const char* text, size_t limit) {
	std::string out;
	if (text == nullptr) return out;
	for (const char* c = text; *c != '\0' && out.size() < limit; ++c) {
		out += (*c == '\n' || *c == '\r' || *c == '"') ? ' ' : *c;
	}
	return out;
}

// "module+0xoffset" for an address inside a loaded module, else "unknown".
std::string ModuleAndOffset(uint64_t address) {
	HMODULE module = nullptr;
	if (address == 0 ||
	    !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                        reinterpret_cast<LPCSTR>(address), &module) ||
	    module == nullptr) {
		return "unknown";
	}
	char path[MAX_PATH] = {};
	if (GetModuleFileNameA(module, path, MAX_PATH) == 0) return "unknown";
	const char* name = std::strrchr(path, '\\');
	name             = name != nullptr ? name + 1 : path;
	return fmt::format("{}+0x{:x}", name, address - reinterpret_cast<uint64_t>(module));
}

std::string ThreadName(uint32_t tid) {
	try {
		for (const auto& [id, name]: LocalPlatform::ProcessThreads()) {
			if (id == tid) return name;
		}
	} catch (...) {
	}
	return {};
}

// Writes the minidump (threads' info and the memory their stacks point at: never the guest's ~12 GB) and the line.
void Record(const char* kind, uint32_t code, uint64_t pc, const char* message, EXCEPTION_POINTERS* pointers) {
	if (g_recorded.exchange(true)) return;

	const char* dir_env = std::getenv("DES_CRASH_DIR");
	std::string dir     = dir_env != nullptr && *dir_env != '\0' ? dir_env : ".";
	SYSTEMTIME  now {};
	GetLocalTime(&now);
	const std::string file = fmt::format("crash-{:04}{:02}{:02}-{:02}{:02}{:02}-{}.dmp", now.wYear, now.wMonth, now.wDay,
	                                     now.wHour, now.wMinute, now.wSecond, GetCurrentProcessId());
	const std::string path = dir + "\\" + file;

	std::string written = "none";
	if (HMODULE dbghelp = LoadLibraryA("dbghelp.dll"); dbghelp != nullptr) {
		const auto write = reinterpret_cast<MiniDumpWriteDumpFn>(GetProcAddress(dbghelp, "MiniDumpWriteDump"));
		HANDLE     out   = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (write != nullptr && out != INVALID_HANDLE_VALUE) {
			MINIDUMP_EXCEPTION_INFORMATION exception {};
			exception.ThreadId          = GetCurrentThreadId();
			exception.ExceptionPointers = pointers;
			exception.ClientPointers    = FALSE;
			const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory |
			                                             MiniDumpWithUnloadedModules);
			if (write(GetCurrentProcess(), GetCurrentProcessId(), out, type, pointers != nullptr ? &exception : nullptr, nullptr,
			          nullptr)) {
				written = file;
			}
		}
		if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
	}

	const uint32_t tid = GetCurrentThreadId();
	EmitLine(fmt::format("DES_CRASH t={} kind={} code=0x{:x} pc=0x{:x} module={} thread={} name={} dump={} message=\"{}\"\n",
	                     ElapsedMilliseconds(), kind, code, pc, ModuleAndOffset(pc), tid, ThreadName(tid), written,
	                     OneLine(message, 400)));
}

bool Contains(const std::string& text, const char* needle) {
	return text.find(needle) != std::string::npos;
}

// Common::ReportCrash: the engine is ending with EXIT (message "<text> in <file>:<line>").
void EngineExitHook(const char* message) {
	if (t_noted.valid) {
		EXCEPTION_RECORD record {};
		record.ExceptionCode    = t_noted.code;
		record.ExceptionAddress = reinterpret_cast<PVOID>(t_noted.pc);
		EXCEPTION_POINTERS pointers {&record, &t_noted.context};
		Record("exception", t_noted.code, t_noted.pc, message, &pointers);
		return;
	}
	std::string text = message != nullptr ? message : "";
	const bool  lost = Contains(text, "DEVICE_LOST") || Contains(text, "device lost") || Contains(text, "Device lost") ||
	                  Contains(text, "device loss");
	Record(lost ? "device_lost" : "exit", 321, reinterpret_cast<uint64_t>(_ReturnAddress()), message, nullptr);
}

LONG WINAPI UnhandledFilter(EXCEPTION_POINTERS* pointers) {
	const auto* record = pointers->ExceptionRecord;
	Record("exception", record->ExceptionCode, reinterpret_cast<uint64_t>(record->ExceptionAddress),
	       "unhandled exception (no handler took it)", pointers);
	return EXCEPTION_CONTINUE_SEARCH;
}

void AbortHandler(int) {
	Record("exit", 3, reinterpret_cast<uint64_t>(_ReturnAddress()), "abort() or std::terminate", nullptr);
	TerminateProcess(GetCurrentProcess(), 3);
}

} // namespace

void Install() {
	static std::atomic_bool installed {false};
	if (installed.exchange(true)) return;
	Common::SetCrashHook(&EngineExitHook);
	SetUnhandledExceptionFilter(&UnhandledFilter);
	std::signal(SIGABRT, &AbortHandler);
	std::set_terminate([] {
		Record("exit", 3, reinterpret_cast<uint64_t>(_ReturnAddress()), "std::terminate (uncaught C++ exception)", nullptr);
		TerminateProcess(GetCurrentProcess(), 3);
	});
}

void NoteException(const void* native_context, uint32_t code, uint64_t pc) {
	if (native_context == nullptr) return;
	t_noted.context = *static_cast<const CONTEXT*>(native_context);
	t_noted.code    = code;
	t_noted.pc      = pc;
	t_noted.valid   = true;
}

} // namespace Desport::Crash

#else

namespace Desport::Crash {
void Install() {}
void NoteException(const void*, uint32_t, uint64_t) {}
} // namespace Desport::Crash

#endif
