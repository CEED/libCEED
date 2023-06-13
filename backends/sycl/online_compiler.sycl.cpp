//==----------- online_compiler.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cstring>
#include <string>
#include <dlfcn.h>

#include <sycl/sycl.hpp>
#include "ocloc_api.h"
#include "online_compiler.hpp"

namespace sycl {
namespace ext::libceed {

void *loadOsLibrary(const std::string &PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  void *so = dlopen(PluginPath.c_str(), RTLD_NOW);
  if (!so) {
    char *Error = dlerror();
    std::cerr << "dlopen(" << PluginPath << ") failed with <" << (Error ? Error : "unknown error") << ">" << std::endl;
  }
  return so;
}

// int unloadOsPluginLibrary(void *Library) {
//   // The mock plugin does not have an associated library, so we allow nullptr
//   // here to avoid it trying to free a non-existent library.
//   if (!Library)
//     return 0;
//   return dlclose(Library);
// }

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) { return dlsym(Library, FunctionName.c_str()); }

static std::vector<const char *> prepareOclocArgs(sycl::info::device_type DeviceType, device_arch DeviceArch, bool Is64Bit,
                                                  const std::string &DeviceStepping, const std::string &UserArgs) {
  std::vector<const char *> Args = {"ocloc", "-q", "-spv_only", "-device"};

  if (DeviceType == sycl::info::device_type::gpu) {
    switch (DeviceArch) {
      case device_arch::gpu_gen9:
        Args.push_back("skl");
        break;

      case device_arch::gpu_gen9_5:
        Args.push_back("cfl");
        break;

      case device_arch::gpu_gen11:
        Args.push_back("icllp");
        break;

      case device_arch::gpu_gen12:
        Args.push_back("tgllp");
        break;

      default:
        Args.push_back("pvc");
    }
  } else {
    // TODO: change that to generic device when ocloc adds support for it.
    // For now "tgllp" is used as the option supported on all known GPU RT.
    Args.push_back("pvc");
  }

  if (DeviceStepping != "") {
    Args.push_back("-revision_id");
    Args.push_back(DeviceStepping.c_str());
  }

  Args.push_back(Is64Bit ? "-64" : "-32");

  if (UserArgs != "") {
    Args.push_back("-options");
    Args.push_back(UserArgs.c_str());
  }

  return Args;
}

/// Compiles the given source \p Source to SPIR-V IL and returns IL as a vector
/// of bytes.
/// @param Source - Either OpenCL or CM source code.
/// @param DeviceType - SYCL device type, e.g. cpu, gpu, accelerator, etc.
/// @param DeviceArch - More detailed info on the target device architecture.
/// @param Is64Bit - If set to true, specifies the 64-bit architecture.
///                  Otherwise, 32-bit is assumed.
/// @param DeviceStepping - implementation specific target device stepping.
/// @param CompileToSPIRVHandle - Output parameter. It is set to the address
///                               of the library function doing the compilation.
/// @param FreeSPIRVOutputsHandle - Output parameter. It is set to the address
///                                 of the library function freeing memory
///                                 allocated during the compilation.
/// @param UserArgs - User's options to ocloc compiler.
static std::vector<byte> compileToSPIRV(const std::string &Source, sycl::info::device_type DeviceType, device_arch DeviceArch, bool Is64Bit,
                                        const std::string &DeviceStepping, void *&CompileToSPIRVHandle, void *&FreeSPIRVOutputsHandle,
                                        const std::vector<std::string> &UserArgs) {
  if (!CompileToSPIRVHandle) {
#ifdef __SYCL_RT_OS_WINDOWS
    static const std::string OclocLibraryName = "ocloc64.dll";
#else
    static const std::string OclocLibraryName = "libocloc.so";
#endif
    void *OclocLibrary = loadOsLibrary(OclocLibraryName);
    if (!OclocLibrary) throw online_compile_error("Cannot load ocloc library: " + OclocLibraryName);
    void *OclocVersionHandle = getOsLibraryFuncAddress(OclocLibrary, "oclocVersion");
    // The initial versions of ocloc library did not have the oclocVersion()
    // function. Those versions had the same API as the first version of ocloc
    // library having that oclocVersion() function.
    int LoadedVersion = ocloc_version_t::OCLOC_VERSION_1_0;
    if (OclocVersionHandle) {
      decltype(::oclocVersion) *OclocVersionFunc = reinterpret_cast<decltype(::oclocVersion) *>(OclocVersionHandle);
      LoadedVersion                              = OclocVersionFunc();
    }
    // The loaded library with version (A.B) is compatible with expected API/ABI
    // version (X.Y) used here if A == B and B >= Y.
    int LoadedVersionMajor  = LoadedVersion >> 16;
    int LoadedVersionMinor  = LoadedVersion & 0xffff;
    int CurrentVersionMajor = ocloc_version_t::OCLOC_VERSION_CURRENT >> 16;
    int CurrentVersionMinor = ocloc_version_t::OCLOC_VERSION_CURRENT & 0xffff;
    if (LoadedVersionMajor != CurrentVersionMajor || LoadedVersionMinor < CurrentVersionMinor)
      throw online_compile_error(std::string("Found incompatible version of ocloc library: (") + std::to_string(LoadedVersionMajor) + "." +
                                 std::to_string(LoadedVersionMinor) + "). The supported versions are (" + std::to_string(CurrentVersionMajor) +
                                 ".N), where (N >= " + std::to_string(CurrentVersionMinor) + ").");

    CompileToSPIRVHandle = getOsLibraryFuncAddress(OclocLibrary, "oclocInvoke");
    if (!CompileToSPIRVHandle) throw online_compile_error("Cannot load oclocInvoke() function");
    FreeSPIRVOutputsHandle = getOsLibraryFuncAddress(OclocLibrary, "oclocFreeOutput");
    if (!FreeSPIRVOutputsHandle) throw online_compile_error("Cannot load oclocFreeOutput() function");
  }

  std::string CombinedUserArgs;
  for (auto UserArg : UserArgs) {
    if (UserArg == "") continue;
    if (CombinedUserArgs != "") CombinedUserArgs = CombinedUserArgs + " " + UserArg;
    else CombinedUserArgs = UserArg;
  }
  std::vector<const char *> Args = prepareOclocArgs(DeviceType, DeviceArch, Is64Bit, DeviceStepping, CombinedUserArgs);

  uint32_t  NumOutputs    = 0;
  byte    **Outputs       = nullptr;
  uint64_t *OutputLengths = nullptr;
  char    **OutputNames   = nullptr;

  const byte    *Sources[]       = {reinterpret_cast<const byte *>(Source.c_str())};
  const char    *SourceName      = "main.cl";
  const uint64_t SourceLengths[] = {Source.length() + 1};

  Args.push_back("-file");
  Args.push_back(SourceName);

  decltype(::oclocInvoke) *OclocInvokeFunc = reinterpret_cast<decltype(::oclocInvoke) *>(CompileToSPIRVHandle);
  int CompileError = OclocInvokeFunc(Args.size(), Args.data(), 1, Sources, SourceLengths, &SourceName, 0, nullptr, nullptr, nullptr, &NumOutputs,
                                     &Outputs, &OutputLengths, &OutputNames);

  std::vector<byte> SpirV;
  std::string       CompileLog;
  for (uint32_t I = 0; I < NumOutputs; I++) {
    size_t NameLen = strlen(OutputNames[I]);
    if (NameLen >= 4 && strstr(OutputNames[I], ".spv") != nullptr && Outputs[I] != nullptr) {
      assert(SpirV.size() == 0 && "More than one SPIR-V output found.");
      SpirV = std::vector<byte>(Outputs[I], Outputs[I] + OutputLengths[I]);
    } else if (!strcmp(OutputNames[I], "stdout.log")) {
      CompileLog = std::string(reinterpret_cast<const char *>(Outputs[I]));
    }
  }

  // Try to free memory before reporting possible error.
  decltype(::oclocFreeOutput) *OclocFreeOutputFunc = reinterpret_cast<decltype(::oclocFreeOutput) *>(FreeSPIRVOutputsHandle);
  int                          MemFreeError        = OclocFreeOutputFunc(&NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  if (CompileError) throw online_compile_error("ocloc reported compilation errors: {\n" + CompileLog + "\n}");
  if (SpirV.empty()) throw online_compile_error("Unexpected output: ocloc did not return SPIR-V");
  if (MemFreeError) throw online_compile_error("ocloc cannot safely free resources");

  return SpirV;
}

template <>
template <>
std::vector<byte> online_compiler<source_language::opencl_c>::compile(const std::string &Source, const std::vector<std::string> &UserArgs) {
  if (OutputFormatVersion != std::pair<int, int>{0, 0}) {
    std::string Version = std::to_string(OutputFormatVersion.first) + ", " + std::to_string(OutputFormatVersion.second);
    throw online_compile_error(std::string("The output format version (") + Version + ") is not supported yet");
  }

  return compileToSPIRV(Source, DeviceType, DeviceArch, Is64Bit, DeviceStepping, CompileToSPIRVHandle, FreeSPIRVOutputsHandle, UserArgs);
}

template <>
template <>
std::vector<byte> online_compiler<source_language::cm>::compile(const std::string &Source, const std::vector<std::string> &UserArgs) {
  if (OutputFormatVersion != std::pair<int, int>{0, 0}) {
    std::string Version = std::to_string(OutputFormatVersion.first) + ", " + std::to_string(OutputFormatVersion.second);
    throw online_compile_error(std::string("The output format version (") + Version + ") is not supported yet");
  }

  std::vector<std::string> CMUserArgs = UserArgs;
  CMUserArgs.push_back("-cmc");
  return compileToSPIRV(Source, DeviceType, DeviceArch, Is64Bit, DeviceStepping, CompileToSPIRVHandle, FreeSPIRVOutputsHandle, CMUserArgs);
}

}  // namespace ext::libceed
}  // namespace sycl
