cmake_minimum_required(VERSION 3.24)

project(TrSSRuntime LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "The first TrSS live runtime requires Windows and the MSVC Windows SDK.")
endif()

add_library(trss_runtime SHARED
  src/trss_runtime.cpp
)

target_compile_features(trss_runtime PRIVATE cxx_std_17)
target_compile_definitions(trss_runtime PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
target_compile_options(trss_runtime PRIVATE
  /W4
  /permissive-
  /EHsc
  /utf-8
)
target_link_libraries(trss_runtime PRIVATE
  d3d11
  dxgi
  d3dcompiler
  d2d1
  dwrite
  comdlg32
  user32
  gdi32
)
set_target_properties(trss_runtime PROPERTIES
  OUTPUT_NAME "dxgi"
  PREFIX ""
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

message(STATUS "TrSS first-party runtime: DXGI app-local shim / D3D11 post-processing")
message(STATUS "Initial scope: compatible D3D11 swap chains with single-sample BGRA/RGBA swap buffers")
message(STATUS "D3D12, D3D9, OpenGL, Vulkan, and protected anti-cheat sessions are not supported by this target")
