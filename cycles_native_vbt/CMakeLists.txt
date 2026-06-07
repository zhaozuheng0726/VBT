add_executable(vbt_sampler_cpu_probe
    vbt_sampler_cpu_probe.cpp
    vbt_cycles_loader.cpp
    vbt_cycles_blob.cpp
)

target_compile_features(vbt_sampler_cpu_probe PRIVATE cxx_std_17)
target_include_directories(vbt_sampler_cpu_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
