cmake_minimum_required(VERSION 3.25)
project(TrSSSampleRecorder LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "TrSS Sample Recorder is a native Windows application.")
endif()

add_executable(TrSSSampleRecorder WIN32
    src/main.cpp
)

target_compile_features(TrSSSampleRecorder PRIVATE cxx_std_17)
target_compile_definitions(TrSSSampleRecorder PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    WINVER=0x0A00
)
target_link_libraries(TrSSSampleRecorder PRIVATE
    advapi32
    comctl32
    gdiplus
    ole32
    psapi
    shell32
)

if(MSVC)
    set_property(TARGET TrSSSampleRecorder PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(TrSSSampleRecorder PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(TrSSSampleRecorder PRIVATE /OPT:REF /OPT:ICF)
endif()

set_target_properties(TrSSSampleRecorder PROPERTIES
    OUTPUT_NAME "TrSSSampleRecorder"
    WIN32_EXECUTABLE TRUE
)
