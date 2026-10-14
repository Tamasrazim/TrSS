// TrSS Direct3D 9 API bring-up for older games.
//
// D3D9 has no compute-shader API. This probe validates the D3D9 device,
// render-target clear, and surface readback path only; neural work needs a
// separate compute backend and a safely shared/copyable frame source.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d9.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace {
struct AdapterInfo {
    UINT ordinal = 0;
    std::string name;
    UINT vendor_id = 0;
    UINT device_id = 0;
    bool hal_available = false;
    std::string reason;
};

void check_hresult(HRESULT result, const char* operation) {
    if (SUCCEEDED(result)) return;
    std::ostringstream message;
    message << operation << " failed (HRESULT 0x"
            << std::hex << std::uppercase
            << static_cast<unsigned long>(static_cast<std::uint32_t>(result)) << ").";
    throw std::runtime_error(message.str());
}

std::string quote_json(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch) << std::dec << std::setfill(' ');
            } else out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

std::string hex_id(UINT value) {
    std::ostringstream out;
    out << "0x" << std::uppercase << std::hex << std::setw(4)
        << std::setfill('0') << value;
    return out.str();
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::vector<AdapterInfo> enumerate_adapters(IDirect3D9* d3d9) {
    std::vector<AdapterInfo> adapters;
    for (UINT ordinal = 0; ordinal < d3d9->GetAdapterCount(); ++ordinal) {
        D3DADAPTER_IDENTIFIER9 identifier{};
        const HRESULT identity_result = d3d9->GetAdapterIdentifier(
            ordinal, 0, &identifier);
        if (FAILED(identity_result)) continue;

        AdapterInfo item;
        item.ordinal = ordinal;
        item.name = identifier.Description;
        item.vendor_id = identifier.VendorId;
        item.device_id = identifier.DeviceId;

        D3DDISPLAYMODE display_mode{};
        const HRESULT mode_result = d3d9->GetAdapterDisplayMode(ordinal, &display_mode);
        if (SUCCEEDED(mode_result)) {
            item.hal_available = SUCCEEDED(d3d9->CheckDeviceType(
                ordinal, D3DDEVTYPE_HAL, display_mode.Format,
                display_mode.Format, TRUE));
        }
        if (!item.hal_available) {
            item.reason = "D3D9 HAL could not confirm a windowed desktop-format device.";
        }
        adapters.push_back(std::move(item));
    }
    return adapters;
}

void print_devices(const std::vector<AdapterInfo>& adapters) {
    std::cout << "{\n  \"api\": \"Direct3D 9\",\n  \"devices\": [";
    if (!adapters.empty()) std::cout << '\n';
    for (std::size_t i = 0; i < adapters.size(); ++i) {
        const auto& item = adapters[i];
        std::cout << "    {\n"
                  << "      \"name\": " << quote_json(item.name) << ",\n"
                  << "      \"adapter_ordinal\": " << item.ordinal << ",\n"
                  << "      \"vendor_id\": " << quote_json(hex_id(item.vendor_id)) << ",\n"
                  << "      \"device_id\": " << quote_json(hex_id(item.device_id)) << ",\n"
                  << "      \"windowed_hal_supported\": " << (item.hal_available ? "true" : "false");
        if (!item.reason.empty()) std::cout << ",\n      \"reason\": " << quote_json(item.reason);
        std::cout << "\n    }" << (i + 1 == adapters.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n"
              << "  \"note\": \"Enumeration is not frame capture or neural-rendering support.\"\n}\n";
}

std::size_t select_adapter(
    const std::vector<AdapterInfo>& adapters, const std::string& requested) {
    const std::string needle = lowercase(requested);
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < adapters.size(); ++i) {
        const auto& item = adapters[i];
        if (!item.hal_available) continue;
        const auto name = lowercase(item.name);
        if ((!needle.empty() && name.find(needle) != std::string::npos)
            || (needle.empty() && name.find("arc") != std::string::npos)) {
            matches.push_back(i);
        }
    }
    if (matches.size() == 1) return matches.front();
    if (matches.empty()) {
        throw std::runtime_error(
            "No unique Direct3D 9 HAL-capable Arc adapter matched. "
            "Use --list-devices and optionally specify --device-substring.");
    }
    throw std::runtime_error("Adapter name matched multiple devices; provide a longer substring.");
}

class ScopedWindow {
public:
    explicit ScopedWindow(HWND window) noexcept : window_(window) {}
    ~ScopedWindow() {
        if (window_ != nullptr) DestroyWindow(window_);
    }
    ScopedWindow(const ScopedWindow&) = delete;
    ScopedWindow& operator=(const ScopedWindow&) = delete;
    HWND get() const noexcept { return window_; }
private:
    HWND window_ = nullptr;
};

HWND create_hidden_window() {
    constexpr wchar_t class_name[] = L"TrSSDirect3D9ProbeWindow";
    const HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.lpfnWndProc = DefWindowProcW;
    window_class.hInstance = instance;
    window_class.lpszClassName = class_name;
    if (RegisterClassExW(&window_class) == 0
        && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        throw std::runtime_error("Could not register the hidden D3D9 probe window.");
    }
    HWND window = CreateWindowExW(
        0, class_name, L"TrSS Direct3D 9 probe", WS_OVERLAPPED,
        CW_USEDEFAULT, CW_USEDEFAULT, 64, 64,
        nullptr, nullptr, instance, nullptr);
    if (window == nullptr) throw std::runtime_error("Could not create the hidden D3D9 probe window.");
    return window;
}

void execute_render_target_roundtrip(IDirect3D9* d3d9, const AdapterInfo& adapter) {
    ScopedWindow window(create_hidden_window());
    D3DPRESENT_PARAMETERS present{};
    present.BackBufferWidth = 64;
    present.BackBufferHeight = 64;
    present.BackBufferFormat = D3DFMT_UNKNOWN;
    present.BackBufferCount = 1;
    present.MultiSampleType = D3DMULTISAMPLE_NONE;
    present.SwapEffect = D3DSWAPEFFECT_DISCARD;
    present.hDeviceWindow = window.get();
    present.Windowed = TRUE;
    present.EnableAutoDepthStencil = FALSE;
    present.FullScreen_RefreshRateInHz = 0;
    present.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;

    ComPtr<IDirect3DDevice9> device;
    HRESULT result = d3d9->CreateDevice(
        adapter.ordinal, D3DDEVTYPE_HAL, window.get(),
        D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE,
        &present, &device);
    if (FAILED(result)) {
        device.Reset();
        result = d3d9->CreateDevice(
            adapter.ordinal, D3DDEVTYPE_HAL, window.get(),
            D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE,
            &present, &device);
    }
    check_hresult(result, "IDirect3D9::CreateDevice");

    constexpr D3DCOLOR clear_color = D3DCOLOR_ARGB(255, 23, 45, 67);
    check_hresult(device->Clear(0, nullptr, D3DCLEAR_TARGET, clear_color, 1.0f, 0),
        "IDirect3DDevice9::Clear");

    ComPtr<IDirect3DSurface9> back_buffer;
    check_hresult(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back_buffer),
        "IDirect3DDevice9::GetBackBuffer");

    D3DSURFACE_DESC desc{};
    check_hresult(back_buffer->GetDesc(&desc), "IDirect3DSurface9::GetDesc");
    if (desc.Format != D3DFMT_X8R8G8B8 && desc.Format != D3DFMT_A8R8G8B8) {
        throw std::runtime_error(
            "The desktop back-buffer is not X8R8G8B8/A8R8G8B8; pixel validation "
            "is intentionally not guessed for an unknown format.");
    }

    ComPtr<IDirect3DSurface9> readback;
    check_hresult(d3d9->CreateOffscreenPlainSurface(
        desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &readback, nullptr),
        "IDirect3D9::CreateOffscreenPlainSurface");
    check_hresult(device->GetRenderTargetData(back_buffer.Get(), readback.Get()),
        "IDirect3DDevice9::GetRenderTargetData");

    D3DLOCKED_RECT locked{};
    check_hresult(readback->LockRect(&locked, nullptr, D3DLOCK_READONLY),
        "IDirect3DSurface9::LockRect");
    if (locked.Pitch <= 0 || locked.pBits == nullptr) {
        readback->UnlockRect();
        throw std::runtime_error("D3D9 readback surface returned an invalid pixel buffer.");
    }

    const auto* row = static_cast<const std::uint8_t*>(locked.pBits)
        + static_cast<std::size_t>(locked.Pitch) * (desc.Height / 2U);
    const auto* pixel = reinterpret_cast<const std::uint32_t*>(row) + desc.Width / 2U;
    const std::uint32_t actual = *pixel & 0x00FFFFFFU;
    const std::uint32_t expected = clear_color & 0x00FFFFFFU;
    readback->UnlockRect();
    if (actual != expected) {
        std::ostringstream message;
        message << "D3D9 render-target readback mismatch: expected 0x"
                << std::hex << expected << ", received 0x" << actual << ".";
        throw std::runtime_error(message.str());
    }
}

void usage() {
    std::cout << "TrSS Direct3D 9 API probe\n"
                 "  trss_d3d9_probe.exe --list-devices\n"
                 "  trss_d3d9_probe.exe --self-test [--device-substring NAME]\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool list = false;
        bool self_test = false;
        std::string needle;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--list-devices") { list = true; continue; }
            if (arg == "--self-test") { self_test = true; continue; }
            if (arg == "--device-substring" && i + 1 < argc) {
                needle = argv[++i];
                if (needle.empty()) throw std::runtime_error("--device-substring cannot be blank.");
                continue;
            }
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (list && self_test) throw std::runtime_error("Choose one operation.");
        if (!needle.empty() && !self_test) {
            throw std::runtime_error("--device-substring is only valid with --self-test.");
        }

        ComPtr<IDirect3D9> d3d9;
        IDirect3D9* raw_d3d9 = Direct3DCreate9(D3D_SDK_VERSION);
        if (raw_d3d9 == nullptr) {
            throw std::runtime_error(
                "Direct3DCreate9 could not create the legacy Direct3D 9 runtime object.");
        }
        d3d9.Attach(raw_d3d9);
        const auto adapters = enumerate_adapters(d3d9.Get());
        if (list) { print_devices(adapters); return 0; }

        const auto index = select_adapter(adapters, needle);
        const auto& adapter = adapters[index];
        execute_render_target_roundtrip(d3d9.Get(), adapter);
        std::cout << "{\n  \"api\": \"Direct3D 9\",\n"
                  << "  \"backend\": \"Direct3D 9 render-target/readback (hardware adapter)\",\n"
                  << "  \"status\": \"passed\",\n"
                  << "  \"adapter\": " << quote_json(adapter.name) << ",\n"
                  << "  \"vendor_id\": " << quote_json(hex_id(adapter.vendor_id)) << ",\n"
                  << "  \"device_id\": " << quote_json(hex_id(adapter.device_id)) << ",\n"
                  << "  \"test\": \"device_clear_render_target_readback\",\n"
                  << "  \"note\": \"D3D9 does not provide compute shaders; this does not run the TrSS model or capture a game.\"\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TrSS Direct3D 9 error: " << error.what() << '\n';
        return 2;
    }
}
