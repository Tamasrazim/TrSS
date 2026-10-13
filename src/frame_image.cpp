#include "frame_image.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

bool FrameImage::valid() const noexcept {
    if (width == 0 || height == 0) {
        return false;
    }

    const auto pixelCount = static_cast<std::uint64_t>(width) * height;
    return pixelCount <= std::numeric_limits<std::size_t>::max() / 4 &&
           bgra.size() == static_cast<std::size_t>(pixelCount * 4);
}

void FrameImage::clear() noexcept {
    width = 0;
    height = 0;
    bgra.clear();
}

bool LoadFrameImage(const std::wstring& path, FrameImage& output,
                    std::wstring& error) {
    output.clear();
    error.clear();

    ComPtr<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(factory.GetAddressOf())
    );
    if (FAILED(result)) {
        error = L"Windows Imaging Component could not be initialized.";
        return false;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    result = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        decoder.GetAddressOf()
    );
    if (FAILED(result)) {
        error = L"The file could not be opened by a Windows image codec.";
        return false;
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    result = decoder->GetFrame(0, frame.GetAddressOf());
    if (FAILED(result)) {
        error = L"The file did not contain a readable image frame.";
        return false;
    }

    UINT width = 0;
    UINT height = 0;
    result = frame->GetSize(&width, &height);
    if (FAILED(result) || width == 0 || height == 0) {
        error = L"The image has invalid dimensions.";
        return false;
    }

    // Refuse absurd dimensions before allocating memory. The limit is for
    // this inspection tool, not a limit imposed by the Windows image codec.
    constexpr std::uint64_t maxPixels = 80'000'000;
    const auto pixelCount = static_cast<std::uint64_t>(width) * height;
    if (width > 16'384 || height > 16'384 || pixelCount > maxPixels ||
        width > std::numeric_limits<UINT>::max() / 4) {
        error = L"The image is too large for this inspection build (limit: 80 megapixels).";
        return false;
    }

    ComPtr<IWICFormatConverter> converter;
    result = factory->CreateFormatConverter(converter.GetAddressOf());
    if (FAILED(result)) {
        error = L"Could not create the Windows image format converter.";
        return false;
    }

    result = converter->Initialize(
        frame.Get(),
        GUID_WICPixelFormat32bppBGRA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom
    );
    if (FAILED(result)) {
        error = L"This image format could not be converted to BGRA pixels.";
        return false;
    }

    FrameImage decoded;
    decoded.width = width;
    decoded.height = height;

    try {
        decoded.bgra.resize(static_cast<std::size_t>(pixelCount) * 4);
    } catch (const std::bad_alloc&) {
        error = L"Not enough memory to load this image.";
        return false;
    }

    const UINT stride = width * 4;
    const UINT bufferSize = static_cast<UINT>(decoded.bgra.size());
    result = converter->CopyPixels(
        nullptr,
        stride,
        bufferSize,
        decoded.bgra.data()
    );
    if (FAILED(result)) {
        error = L"Windows could not decode the image pixels.";
        return false;
    }

    output = std::move(decoded);
    return true;
}

FrameImage MakeDetailReference(const FrameImage& source, float strength) {
    if (!source.valid()) {
        return {};
    }

    FrameImage output;
    output.width = source.width;
    output.height = source.height;
    output.bgra.resize(source.bgra.size());

    // Keep the amount predictable and prevent extreme slider values from
    // creating overflow or turning the comparison into a harsh edge effect.
    strength = std::clamp(strength, 0.0f, 2.0f);

    const int width = static_cast<int>(source.width);
    const int height = static_cast<int>(source.height);

    for (int y = 0; y < height; ++y) {
        const int northY = (y > 0) ? y - 1 : y;
        const int southY = (y + 1 < height) ? y + 1 : y;

        for (int x = 0; x < width; ++x) {
            const int westX = (x > 0) ? x - 1 : x;
            const int eastX = (x + 1 < width) ? x + 1 : x;

            const std::size_t center =
                (static_cast<std::size_t>(y) * source.width +
                 static_cast<std::size_t>(x)) * 4;
            const std::size_t north =
                (static_cast<std::size_t>(northY) * source.width +
                 static_cast<std::size_t>(x)) * 4;
            const std::size_t south =
                (static_cast<std::size_t>(southY) * source.width +
                 static_cast<std::size_t>(x)) * 4;
            const std::size_t west =
                (static_cast<std::size_t>(y) * source.width +
                 static_cast<std::size_t>(westX)) * 4;
            const std::size_t east =
                (static_cast<std::size_t>(y) * source.width +
                 static_cast<std::size_t>(eastX)) * 4;

            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float original = source.bgra[center + channel];
                const float blur =
                    (4.0f * original +
                     source.bgra[north + channel] +
                     source.bgra[south + channel] +
                     source.bgra[west + channel] +
                     source.bgra[east + channel]) / 8.0f;

                const float sharpened = original + strength * (original - blur);
                output.bgra[center + channel] = static_cast<std::uint8_t>(
                    std::clamp(std::lround(sharpened), 0L, 255L)
                );
            }

            // The reference pass never invents transparency.
            output.bgra[center + 3] = source.bgra[center + 3];
        }
    }

    return output;
}
