#pragma once

#include <cstdint>
#include <string>
#include <vector>

// An image is stored top-to-bottom as 8-bit BGRA pixels. Keeping this
// representation explicit makes the Win32 display path predictable and
// leaves room for a later GPU upload path.
struct FrameImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> bgra;

    [[nodiscard]] bool valid() const noexcept;
    void clear() noexcept;
};

// Decode a local image through Windows Imaging Component (WIC). This uses
// Windows' built-in image codecs instead of bundling an image library.
bool LoadFrameImage(const std::wstring& path, FrameImage& output,
                    std::wstring& error);

// Make a conventional unsharp-mask image for A/B inspection. This is only a
// visible baseline for the comparison UI; it is NOT a neural network or a
// replacement for a licensed, trained model.
FrameImage MakeDetailReference(const FrameImage& source, float strength = 1.25f);
