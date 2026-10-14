cmake_minimum_required(VERSION 3.24)

project(TrSSDirectXApiProbes LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "The TrSS DirectX 9/11/12 native probes require Windows.")
endif()
if(NOT MSVC)
  message(FATAL_ERROR "Use a Visual Studio/MSVC toolchain to build this Windows SDK target.")
endif()

add_executable(trss_dx12_probe
  src/trss_dx12_probe.cpp
)

target_compile_features(trss_dx12_probe PRIVATE cxx_std_17)
target_compile_options(trss_dx12_probe PRIVATE
  /W4
  /permissive-
  /EHsc
  /utf-8
)
target_link_libraries(trss_dx12_probe PRIVATE
  d3d12
  dxgi
  d3dcompiler
)

set_target_properties(trss_dx12_probe PROPERTIES
  CXX_EXTENSIONS OFF
  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:trss_dx12_probe>"
)

message(STATUS "TrSS DirectX 9/11/12 native API probes enabled")
message(STATUS "These probes do not run TrSS neural inference or game capture")


# Direct3D 11 has its own compute API and is tested independently of D3D12.
add_executable(trss_d3d11_probe
  src/trss_d3d11_probe.cpp
)
target_compile_features(trss_d3d11_probe PRIVATE cxx_std_17)
target_compile_options(trss_d3d11_probe PRIVATE /W4 /permissive- /EHsc /utf-8)
target_link_libraries(trss_d3d11_probe PRIVATE d3d11 dxgi d3dcompiler)
set_target_properties(trss_d3d11_probe PROPERTIES CXX_EXTENSIONS OFF)

# Direct3D 9 exposes a graphics/render-target path, not compute shaders.
add_executable(trss_d3d9_probe
  src/trss_d3d9_probe.cpp
)
target_compile_features(trss_d3d9_probe PRIVATE cxx_std_17)
target_compile_options(trss_d3d9_probe PRIVATE /W4 /permissive- /EHsc /utf-8)
target_link_libraries(trss_d3d9_probe PRIVATE d3d9 user32)
set_target_properties(trss_d3d9_probe PROPERTIES CXX_EXTENSIONS OFF)

message(STATUS "TrSS API probes: Direct3D 9Ex, Direct3D 11, and Direct3D 12")
