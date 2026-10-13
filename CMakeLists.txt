cmake_minimum_required(VERSION 3.24)

project(XESS5
    VERSION 0.0.1
    DESCRIPTION "Independent local-first neural graphics research for Intel Arc"
    LANGUAGES CXX
)

if(NOT WIN32)
    message(FATAL_ERROR
        "XESS5 currently targets native Windows. Cross-platform support will be added "
        "only after the Windows rendering path is verified."
    )
endif()

add_executable(XESS5 WIN32
    src/main.cpp
    src/frame_image.cpp
)

target_compile_features(XESS5 PRIVATE cxx_std_20)

target_compile_definitions(XESS5 PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    WINVER=0x0A00
    _WIN32_WINNT=0x0A00
)

target_include_directories(XESS5 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# These APIs ship with Windows. This first frame-inspection path needs no
# separate image viewer, runtime installer, or third-party GUI framework.
target_link_libraries(XESS5 PRIVATE
    windowscodecs
    ole32
    comdlg32
    gdi32
    user32
)

if(MSVC)
    target_compile_options(XESS5 PRIVATE
        /W4
        /permissive-
        /EHsc
        /Zc:__cplusplus
    )
endif()
