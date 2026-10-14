cmake_minimum_required(VERSION 3.24)

project(TrSSDirectX12 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "The TrSS Direct3D 12 backend requires Windows.")
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

message(STATUS "TrSS Direct3D 12 compute bring-up target enabled")
message(STATUS "This target verifies native compute execution; it does not run TrSS neural inference")
