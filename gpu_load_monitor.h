/* Optional driver telemetry. No NVML link-time/runtime dependency is required.
   Used only by the CUDA host implementation. GPL-3.0-or-later. */
#ifndef ECM_GPU_LOAD_MONITOR_H
#define ECM_GPU_LOAD_MONITOR_H
#include <cstring>
#include <nvml.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

class ecm_gpu_load_monitor {
#ifdef _WIN32
  HMODULE library = nullptr;
#else
  void *library = nullptr;
#endif
  nvmlDevice_t device = nullptr;
  decltype(&nvmlShutdown) shutdown = nullptr;
  decltype(&nvmlDeviceGetUtilizationRates) utilization = nullptr;
  bool initialized = false;
  template<typename T> T symbol(const char *name) {
#ifdef _WIN32
    return reinterpret_cast<T>(GetProcAddress(library, name));
#else
    return reinterpret_cast<T>(dlsym(library, name));
#endif
  }
public:
  ecm_gpu_load_monitor() = default;
  ecm_gpu_load_monitor(const ecm_gpu_load_monitor &) = delete;
  ecm_gpu_load_monitor &operator=(const ecm_gpu_load_monitor &) = delete;
  void open(const char *pci_bus_id) {
#ifdef _WIN32
    char path[MAX_PATH];
    unsigned length = GetSystemDirectoryA(path, MAX_PATH);
    if(length && length + 10 < MAX_PATH) {
      strcat(path, "\\nvml.dll");
      library = LoadLibraryA(path);
    }
    if(!library) {
      length = GetEnvironmentVariableA("ProgramW6432", path, MAX_PATH);
      const char *suffix = "\\NVIDIA Corporation\\NVSMI\\nvml.dll";
      if(length && length + strlen(suffix) < MAX_PATH) {
        strcat(path, suffix);
        library = LoadLibraryA(path);
      }
    }
#else
    library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if(!library) library = dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if(!library) return;
    auto init = symbol<decltype(&nvmlInit_v2)>("nvmlInit_v2");
    shutdown = symbol<decltype(shutdown)>("nvmlShutdown");
    auto by_bus = symbol<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>("nvmlDeviceGetHandleByPciBusId_v2");
    utilization = symbol<decltype(utilization)>("nvmlDeviceGetUtilizationRates");
    if(!init || !shutdown || !by_bus || !utilization || init() != NVML_SUCCESS) return;
    initialized = true;
    if(by_bus(pci_bus_id, &device) != NVML_SUCCESS) device = nullptr;
  }
  int sample() {
    nvmlUtilization_t value;
    if(!device || utilization(device, &value) != NVML_SUCCESS || value.gpu > 100) return -1;
    return (int)value.gpu;
  }
  ~ecm_gpu_load_monitor() {
    if(initialized) shutdown();
    if(library) {
#ifdef _WIN32
      FreeLibrary(library);
#else
      dlclose(library);
#endif
    }
  }
};
#endif
