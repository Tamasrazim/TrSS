// TrSS first-party Direct3D 11 post-processing runtime.
//
// This is an app-local DXGI shim, not ReShade and not a wrapper around another
// rendering tool. The first target is ordinary D3D11 swap chains created through
// DXGI. D3D12, Vulkan, OpenGL, D3D9, anti-cheat-protected sessions, and games that
// bypass the expected DXGI factory path are outside this initial implementation.
//
// The runtime deliberately uses documented Windows / Direct3D APIs and owns its
// small post-processing shader and menu. It does not read, modify, or bypass
// anti-cheat memory or protections.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <commdlg.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <d2d1.h>
#include <dwrite.h>
#include <windowsx.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "comdlg32.lib")

using Microsoft::WRL::ComPtr;

namespace trss_runtime {

constexpr wchar_t kConfigName[] = L"TrSS.ini";
constexpr wchar_t kPresetFolder[] = L"TrSS\\Presets";
constexpr float kMenuWidth = 520.0f;
constexpr float kMenuHeight = 620.0f;
constexpr float kMenuLeft = 22.0f;
constexpr float kMenuTop = 22.0f;
constexpr int kSliderStartY = 128;
constexpr int kSliderRowHeight = 34;
constexpr int kSliderTrackLeft = 232;
constexpr int kSliderTrackRight = 420;
constexpr int kSliderCount = 10;

struct Settings {
    bool enabled = true;
    float brightness = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    float gamma = 1.0f;
    float sharpen = 0.0f;
    float bloom = 0.0f;
    float vignette = 0.0f;
    float grain = 0.0f;
    float temperature = 0.0f;
    float chromatic = 0.0f;
    std::wstring activePreset = L"Neutral";
};

struct SliderDefinition {
    const wchar_t* label;
    float Settings::*member;
    float minimum;
    float maximum;
    const wchar_t* format;
};

constexpr std::array<SliderDefinition, kSliderCount> kSliders{{
    {L"Brightness", &Settings::brightness, -0.50f, 0.50f, L"%+.2f"},
    {L"Contrast", &Settings::contrast, 0.50f, 1.80f, L"%.2f"},
    {L"Saturation", &Settings::saturation, 0.0f, 2.0f, L"%.2f"},
    {L"Gamma", &Settings::gamma, 0.50f, 2.0f, L"%.2f"},
    {L"Sharpen", &Settings::sharpen, 0.0f, 2.0f, L"%.2f"},
    {L"Bloom", &Settings::bloom, 0.0f, 2.0f, L"%.2f"},
    {L"Vignette", &Settings::vignette, 0.0f, 1.0f, L"%.2f"},
    {L"Film grain", &Settings::grain, 0.0f, 0.08f, L"%.3f"},
    {L"Color temperature", &Settings::temperature, -0.25f, 0.25f, L"%+.3f"},
    {L"Chromatic aberration", &Settings::chromatic, 0.0f, 0.010f, L"%.3f"},
}};

std::mutex g_settingsMutex;
std::mutex g_patchMutex;
std::mutex g_renderMutex;
std::once_flag g_pathsOnce;
std::once_flag g_dxgiOnce;
Settings g_settings;
bool g_settingsLoaded = false;
std::atomic_bool g_menuVisible{false};
std::atomic_bool g_enabled{true};
std::atomic_bool g_insidePresent{false};
int g_scrollOffset = 0;
int g_dragSlider = -1;
HWND g_gameWindow = nullptr;
WNDPROC g_previousWndProc = nullptr;
HMODULE g_module = nullptr;
HMODULE g_realDxgi = nullptr;
std::wstring g_runtimeDirectory;
std::wstring g_configPath;
std::wstring g_presetDirectory;
float g_animationTime = 0.0f;

using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
using CreateFactory2Fn = HRESULT(WINAPI*)(UINT, REFIID, void**);
using DeclareRemovalFn = HRESULT(WINAPI*)();
using DisableVBlankFn = HRESULT(WINAPI*)();
using GetDebugInterfaceFn = HRESULT(WINAPI*)(REFIID, void**);
using GetDebugInterface1Fn = HRESULT(WINAPI*)(UINT, REFIID, void**);
CreateFactoryFn g_realCreateFactory = nullptr;
CreateFactoryFn g_realCreateFactory1 = nullptr;
CreateFactory2Fn g_realCreateFactory2 = nullptr;
DeclareRemovalFn g_realDeclareRemovalSupport = nullptr;
DisableVBlankFn g_realDisableVBlankVirtualization = nullptr;
GetDebugInterfaceFn g_realGetDebugInterface = nullptr;
GetDebugInterface1Fn g_realGetDebugInterface1 = nullptr;

using CreateSwapChainFn = HRESULT(STDMETHODCALLTYPE*)(
    IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
using CreateSwapChainForHwndFn = HRESULT(STDMETHODCALLTYPE*)(
    IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using ResizeBuffersFn = HRESULT(STDMETHODCALLTYPE*)(
    IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
CreateSwapChainFn g_realCreateSwapChain = nullptr;
CreateSwapChainForHwndFn g_realCreateSwapChainForHwnd = nullptr;
PresentFn g_realPresent = nullptr;
ResizeBuffersFn g_realResizeBuffers = nullptr;

struct RuntimeResources {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11Device1> device1;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11DeviceContext1> context1;
    ComPtr<ID3DDeviceContextState> overlayState;
    ComPtr<ID3D11Texture2D> copyTexture;
    ComPtr<ID3D11ShaderResourceView> sourceView;
    ComPtr<ID3D11RenderTargetView> backBufferView;
    ComPtr<ID3D11VertexShader> vertexShader;
    ComPtr<ID3D11PixelShader> pixelShader;
    ComPtr<ID3D11Buffer> parameterBuffer;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11RasterizerState> rasterizer;
    ComPtr<ID3D11BlendState> blend;
    ComPtr<ID3D11DepthStencilState> depthStencil;
    ComPtr<ID2D1Factory> d2dFactory;
    ComPtr<IDWriteFactory> writeFactory;
    ComPtr<ID2D1RenderTarget> d2dTarget;
    ComPtr<ID2D1SolidColorBrush> panelBrush;
    ComPtr<ID2D1SolidColorBrush> panelHeaderBrush;
    ComPtr<ID2D1SolidColorBrush> textBrush;
    ComPtr<ID2D1SolidColorBrush> mutedBrush;
    ComPtr<ID2D1SolidColorBrush> accentBrush;
    ComPtr<ID2D1SolidColorBrush> sliderBackBrush;
    ComPtr<ID2D1SolidColorBrush> buttonBrush;
    ComPtr<IDWriteTextFormat> titleFormat;
    ComPtr<IDWriteTextFormat> textFormat;
    ComPtr<IDWriteTextFormat> smallFormat;
    ComPtr<IDWriteTextFormat> buttonFormat;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    IDXGISwapChain* swapChain = nullptr;
    bool ready = false;
} g_resources;

std::wstring module_directory() {
    wchar_t path[32768]{};
    const DWORD size = GetModuleFileNameW(g_module, path, static_cast<DWORD>(std::size(path)));
    if (size == 0 || size >= std::size(path)) return L".";
    std::wstring full(path, size);
    const auto slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : full.substr(0, slash);
}

std::wstring join_path(const std::wstring& left, const std::wstring& right) {
    if (left.empty()) return right;
    if (left.back() == L'\\' || left.back() == L'/') return left + right;
    return left + L"\\" + right;
}

void ensure_runtime_paths() {
    std::call_once(g_pathsOnce, [] {
        g_runtimeDirectory = module_directory();
        g_configPath = join_path(g_runtimeDirectory, kConfigName);
        g_presetDirectory = join_path(g_runtimeDirectory, kPresetFolder);
    });
}

std::wstring config_value(const wchar_t* section, const wchar_t* key, const wchar_t* fallback) {
    wchar_t buffer[256]{};
    GetPrivateProfileStringW(section, key, fallback, buffer,
        static_cast<DWORD>(std::size(buffer)), g_configPath.c_str());
    return buffer;
}

float clamp_float(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

float config_float(const wchar_t* key, float fallback, float low, float high) {
    const auto text = config_value(L"Effects", key, L"");
    if (text.empty()) return fallback;
    wchar_t* end = nullptr;
    const float parsed = wcstof(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(parsed)) return fallback;
    return clamp_float(parsed, low, high);
}

Settings defaults() { return Settings{}; }

void apply_preset_values(Settings& value, const std::wstring& preset) {
    value = defaults();
    value.activePreset = preset;
    if (_wcsicmp(preset.c_str(), L"Cinematic") == 0) {
        value.contrast = 1.10f; value.saturation = 0.88f; value.temperature = 0.06f;
        value.vignette = 0.18f; value.bloom = 0.22f; value.sharpen = 0.18f;
    } else if (_wcsicmp(preset.c_str(), L"Vibrant") == 0) {
        value.contrast = 1.08f; value.saturation = 1.38f; value.bloom = 0.12f;
        value.sharpen = 0.12f;
    } else if (_wcsicmp(preset.c_str(), L"Clarity") == 0) {
        value.contrast = 1.06f; value.saturation = 1.04f; value.sharpen = 0.72f;
    } else if (_wcsicmp(preset.c_str(), L"Night") == 0) {
        value.brightness = 0.12f; value.contrast = 0.94f; value.gamma = 1.12f;
        value.saturation = 0.92f; value.bloom = 0.08f;
    } else if (_wcsicmp(preset.c_str(), L"Film") == 0) {
        value.contrast = 1.04f; value.saturation = 0.84f; value.temperature = -0.035f;
        value.vignette = 0.22f; value.grain = 0.022f; value.bloom = 0.10f;
    } else {
        value.activePreset = L"Neutral";
    }
}

void save_settings_file() {
    Settings value;
    {
        std::lock_guard<std::mutex> lock(g_settingsMutex);
        value = g_settings;
    }
    WritePrivateProfileStringW(L"TrSSRuntime", L"Product", L"TrSS", g_configPath.c_str());
    WritePrivateProfileStringW(L"TrSSRuntime", L"Version", L"0.1.0", g_configPath.c_str());
    WritePrivateProfileStringW(L"TrSSRuntime", L"ActivePreset", value.activePreset.c_str(), g_configPath.c_str());
    WritePrivateProfileStringW(L"Effects", L"Enabled", value.enabled ? L"1" : L"0", g_configPath.c_str());
    auto writeFloat = [&](const wchar_t* key, float number) {
        wchar_t buffer[64]{};
        swprintf_s(buffer, L"%.6f", static_cast<double>(number));
        WritePrivateProfileStringW(L"Effects", key, buffer, g_configPath.c_str());
    };
    writeFloat(L"Brightness", value.brightness);
    writeFloat(L"Contrast", value.contrast);
    writeFloat(L"Saturation", value.saturation);
    writeFloat(L"Gamma", value.gamma);
    writeFloat(L"Sharpen", value.sharpen);
    writeFloat(L"Bloom", value.bloom);
    writeFloat(L"Vignette", value.vignette);
    writeFloat(L"Grain", value.grain);
    writeFloat(L"Temperature", value.temperature);
    writeFloat(L"Chromatic", value.chromatic);
}

void load_settings_file() {
    ensure_runtime_paths();
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    if (g_settingsLoaded) return;
    g_settings = defaults();
    g_settings.activePreset = config_value(L"TrSSRuntime", L"ActivePreset", L"Neutral");
    const auto product = config_value(L"TrSSRuntime", L"Product", L"");
    if (_wcsicmp(product.c_str(), L"TrSS") != 0) {
        g_settings.activePreset = L"Neutral";
    }
    g_settings.enabled = GetPrivateProfileIntW(L"Effects", L"Enabled", 1, g_configPath.c_str()) != 0;
    g_settings.brightness = config_float(L"Brightness", 0.0f, -0.50f, 0.50f);
    g_settings.contrast = config_float(L"Contrast", 1.0f, 0.50f, 1.80f);
    g_settings.saturation = config_float(L"Saturation", 1.0f, 0.0f, 2.0f);
    g_settings.gamma = config_float(L"Gamma", 1.0f, 0.50f, 2.0f);
    g_settings.sharpen = config_float(L"Sharpen", 0.0f, 0.0f, 2.0f);
    g_settings.bloom = config_float(L"Bloom", 0.0f, 0.0f, 2.0f);
    g_settings.vignette = config_float(L"Vignette", 0.0f, 0.0f, 1.0f);
    g_settings.grain = config_float(L"Grain", 0.0f, 0.0f, 0.08f);
    g_settings.temperature = config_float(L"Temperature", 0.0f, -0.25f, 0.25f);
    g_settings.chromatic = config_float(L"Chromatic", 0.0f, 0.0f, 0.010f);
    g_enabled = g_settings.enabled;
    g_settingsLoaded = true;
}

void assign_float(Settings& to, const Settings& from, const SliderDefinition& definition) {
    to.*(definition.member) = clamp_float(from.*(definition.member), definition.minimum, definition.maximum);
}

void set_preset(const std::wstring& name) {
    {
        std::lock_guard<std::mutex> lock(g_settingsMutex);
        apply_preset_values(g_settings, name);
        g_enabled = g_settings.enabled;
    }
    save_settings_file();
}

std::wstring preset_directory() { return g_presetDirectory; }

bool write_preset_file(const std::wstring& path, const Settings& value) {
    auto writeFloat = [&](const wchar_t* key, float number) {
        wchar_t buffer[64]{};
        swprintf_s(buffer, L"%.6f", static_cast<double>(number));
        return WritePrivateProfileStringW(L"Effects", key, buffer, path.c_str()) != FALSE;
    };
    bool ok = true;
    ok = WritePrivateProfileStringW(L"TrSSPreset", L"Product", L"TrSS", path.c_str()) != FALSE && ok;
    ok = writeFloat(L"Brightness", value.brightness) && ok;
    ok = writeFloat(L"Contrast", value.contrast) && ok;
    ok = writeFloat(L"Saturation", value.saturation) && ok;
    ok = writeFloat(L"Gamma", value.gamma) && ok;
    ok = writeFloat(L"Sharpen", value.sharpen) && ok;
    ok = writeFloat(L"Bloom", value.bloom) && ok;
    ok = writeFloat(L"Vignette", value.vignette) && ok;
    ok = writeFloat(L"Grain", value.grain) && ok;
    ok = writeFloat(L"Temperature", value.temperature) && ok;
    ok = writeFloat(L"Chromatic", value.chromatic) && ok;
    return ok;
}

bool save_preset_dialog(HWND owner) {
    CreateDirectoryW(g_presetDirectory.c_str(), nullptr);
    wchar_t path[MAX_PATH]{};
    wcscpy_s(path, L"Custom.ini");
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = L"TrSS presets (*.ini)\0*.ini\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = path;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrInitialDir = g_presetDirectory.c_str();
    dialog.lpstrDefExt = L"ini";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!GetSaveFileNameW(&dialog)) return false;
    Settings value;
    { std::lock_guard<std::mutex> lock(g_settingsMutex); value = g_settings; }
    return write_preset_file(path, value);
}

bool load_preset_dialog(HWND owner) {
    wchar_t path[MAX_PATH]{};
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = L"TrSS presets (*.ini)\0*.ini\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = path;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrInitialDir = g_presetDirectory.c_str();
    dialog.lpstrDefExt = L"ini";
    dialog.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameW(&dialog)) return false;
    // Read from the selected file directly; never execute or load code from a preset.
    wchar_t productValue[64]{};
    GetPrivateProfileStringW(L"TrSSPreset", L"Product", L"", productValue,
        static_cast<DWORD>(std::size(productValue)), path);
    if (_wcsicmp(productValue, L"TrSS") != 0) {
        MessageBoxW(owner, L"This file is not a TrSS preset.", L"TrSS Runtime", MB_OK | MB_ICONWARNING);
        return false;
    }
    Settings value;
    { std::lock_guard<std::mutex> lock(g_settingsMutex); value = g_settings; }
    auto readFloat = [&](const wchar_t* key, float fallback, float low, float high) {
        wchar_t buffer[64]{};
        GetPrivateProfileStringW(L"Effects", key, L"", buffer, static_cast<DWORD>(std::size(buffer)), path);
        if (!buffer[0]) return fallback;
        wchar_t* end = nullptr;
        const float parsed = wcstof(buffer, &end);
        return end == buffer || !std::isfinite(parsed) ? fallback : clamp_float(parsed, low, high);
    };
    value.brightness = readFloat(L"Brightness", value.brightness, -0.50f, 0.50f);
    value.contrast = readFloat(L"Contrast", value.contrast, 0.50f, 1.80f);
    value.saturation = readFloat(L"Saturation", value.saturation, 0.0f, 2.0f);
    value.gamma = readFloat(L"Gamma", value.gamma, 0.50f, 2.0f);
    value.sharpen = readFloat(L"Sharpen", value.sharpen, 0.0f, 2.0f);
    value.bloom = readFloat(L"Bloom", value.bloom, 0.0f, 2.0f);
    value.vignette = readFloat(L"Vignette", value.vignette, 0.0f, 1.0f);
    value.grain = readFloat(L"Grain", value.grain, 0.0f, 0.08f);
    value.temperature = readFloat(L"Temperature", value.temperature, -0.25f, 0.25f);
    value.chromatic = readFloat(L"Chromatic", value.chromatic, 0.0f, 0.010f);
    value.activePreset = L"Custom";
    {
        std::lock_guard<std::mutex> lock(g_settingsMutex);
        g_settings = value;
        g_enabled = value.enabled;
    }
    save_settings_file();
    return true;
}

HRESULT compile_shader(
    const char* source, const char* target, ComPtr<ID3DBlob>& bytecode) {
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(
        source, strlen(source), "TrSS first-party effect shader", nullptr, nullptr,
        "main", target,
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0, &bytecode, &errors);
    if (FAILED(hr) && errors && errors->GetBufferPointer()) {
        OutputDebugStringA(static_cast<const char*>(errors->GetBufferPointer()));
    }
    return hr;
}

const char* kVertexShader = R"(
struct VSOut { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
VSOut main(uint id : SV_VertexID) {
    VSOut output;
    float2 position = id == 0 ? float2(-1.0, -1.0) :
                      id == 1 ? float2(-1.0, 3.0) : float2(3.0, -1.0);
    output.position = float4(position, 0.0, 1.0);
    output.uv = float2((position.x + 1.0) * 0.5, 1.0 - (position.y + 1.0) * 0.5);
    return output;
})";

const char* kPixelShader = R"(
Texture2D SourceFrame : register(t0);
SamplerState LinearSampler : register(s0);
cbuffer TrSSParams : register(b0) {
    float Brightness; float Contrast; float Saturation; float Gamma;
    float Sharpen; float Bloom; float Vignette; float Grain;
    float Temperature; float Chromatic; float TimeValue; float Enabled;
    float ResolutionX; float ResolutionY; float Pad0; float Pad1;
}
float3 safe_sample(float2 uv) {
    return SourceFrame.Sample(LinearSampler, saturate(uv)).rgb;
}
float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET {
    float2 texel = 1.0 / max(float2(ResolutionX, ResolutionY), float2(1.0, 1.0));
    float3 original;
    if (Chromatic > 0.00001) {
        original.r = safe_sample(uv + float2(Chromatic, 0.0)).r;
        original.g = safe_sample(uv).g;
        original.b = safe_sample(uv - float2(Chromatic, 0.0)).b;
    } else {
        original = safe_sample(uv);
    }
    float3 color = original;
    if (Sharpen > 0.00001 || Bloom > 0.00001) {
        float3 n = safe_sample(uv + float2(0.0, texel.y));
        float3 s = safe_sample(uv - float2(0.0, texel.y));
        float3 e = safe_sample(uv + float2(texel.x, 0.0));
        float3 w = safe_sample(uv - float2(texel.x, 0.0));
        float3 nearby = (n + s + e + w) * 0.25;
        color += (original - nearby) * (Sharpen * 0.85);
        float3 bright = max(nearby - float3(0.68, 0.68, 0.68), 0.0);
        color += bright * Bloom * 0.35;
    }
    float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    color = lerp(luma.xxx, color, Saturation);
    color = (color - 0.5) * Contrast + 0.5 + Brightness;
    color *= float3(1.0 + Temperature, 1.0, 1.0 - Temperature);
    color = pow(saturate(color), 1.0 / max(Gamma, 0.05));
    float2 centered = (uv - 0.5) * float2(ResolutionX / max(ResolutionY, 1.0), 1.0);
    float vignetteFactor = 1.0 - Vignette * smoothstep(0.22, 0.82, length(centered));
    color *= vignetteFactor;
    float2 pixel = uv * float2(ResolutionX, ResolutionY);
    float noise = frac(sin(dot(pixel + TimeValue * 173.0, float2(12.9898, 78.233))) * 43758.5453) - 0.5;
    color += noise * Grain;
    return float4(saturate(color), 1.0);
})";

void release_d2d_resources() {
    g_resources.d2dTarget.Reset();
    g_resources.panelBrush.Reset();
    g_resources.panelHeaderBrush.Reset();
    g_resources.textBrush.Reset();
    g_resources.mutedBrush.Reset();
    g_resources.accentBrush.Reset();
    g_resources.sliderBackBrush.Reset();
    g_resources.buttonBrush.Reset();
    g_resources.titleFormat.Reset();
    g_resources.textFormat.Reset();
    g_resources.smallFormat.Reset();
    g_resources.buttonFormat.Reset();
    g_resources.d2dFactory.Reset();
    g_resources.writeFactory.Reset();
}

void reset_render_resources() {
    release_d2d_resources();
    g_resources.depthStencil.Reset();
    g_resources.blend.Reset();
    g_resources.rasterizer.Reset();
    g_resources.sampler.Reset();
    g_resources.parameterBuffer.Reset();
    g_resources.vertexShader.Reset();
    g_resources.pixelShader.Reset();
    g_resources.backBufferView.Reset();
    g_resources.sourceView.Reset();
    g_resources.copyTexture.Reset();
    g_resources.overlayState.Reset();
    g_resources.context1.Reset();
    g_resources.context.Reset();
    g_resources.device1.Reset();
    g_resources.device.Reset();
    g_resources.width = 0;
    g_resources.height = 0;
    g_resources.format = DXGI_FORMAT_UNKNOWN;
    g_resources.swapChain = nullptr;
    g_resources.ready = false;
}

bool make_d2d_target(ID3D11Texture2D* backBuffer) {
    ComPtr<IDXGISurface> surface;
    if (FAILED(backBuffer->QueryInterface(IID_PPV_ARGS(&surface)))) return false;
    if (FAILED(D2D1CreateFactory(
            D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory), nullptr,
            reinterpret_cast<void**>(g_resources.d2dFactory.GetAddressOf())))) return false;
    if (FAILED(DWriteCreateFactory(
            DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(g_resources.writeFactory.GetAddressOf())))) return false;
    D2D1_RENDER_TARGET_PROPERTIES properties = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_IGNORE),
        96.0f, 96.0f, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_FEATURE_LEVEL_DEFAULT);
    if (FAILED(g_resources.d2dFactory->CreateDxgiSurfaceRenderTarget(
            surface.Get(), &properties, &g_resources.d2dTarget))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x10141B, 0.97f), &g_resources.panelBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x1B222D, 1.0f), &g_resources.panelHeaderBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0xF1F4F8, 1.0f), &g_resources.textBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x96A2B3, 1.0f), &g_resources.mutedBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x88C7FF, 1.0f), &g_resources.accentBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x394454, 1.0f), &g_resources.sliderBackBrush))) return false;
    if (FAILED(g_resources.d2dTarget->CreateSolidColorBrush(
            D2D1::ColorF(0x222B37, 1.0f), &g_resources.buttonBrush))) return false;

    const HRESULT titleHr = g_resources.writeFactory->CreateTextFormat(
        L"Segoe UI Semibold", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 20.0f, L"en-US",
        &g_resources.titleFormat);
    const HRESULT textHr = g_resources.writeFactory->CreateTextFormat(
        L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 12.0f, L"en-US",
        &g_resources.textFormat);
    const HRESULT smallHr = g_resources.writeFactory->CreateTextFormat(
        L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 10.0f, L"en-US",
        &g_resources.smallFormat);
    const HRESULT buttonHr = g_resources.writeFactory->CreateTextFormat(
        L"Segoe UI Semibold", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 11.0f, L"en-US",
        &g_resources.buttonFormat);
    return SUCCEEDED(titleHr) && SUCCEEDED(textHr) && SUCCEEDED(smallHr) && SUCCEEDED(buttonHr);
}

bool initialize_render_resources(
    IDXGISwapChain* swapChain, ID3D11Device* device,
    ID3D11DeviceContext* context, ID3D11Texture2D* backBuffer,
    const D3D11_TEXTURE2D_DESC& description) {
    g_resources.device = device;
    g_resources.context = context;
    g_resources.width = description.Width;
    g_resources.height = description.Height;
    g_resources.format = description.Format;
    g_resources.swapChain = swapChain;

    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&g_resources.device1)))) return false;
    if (FAILED(context->QueryInterface(IID_PPV_ARGS(&g_resources.context1)))) return false;

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL chosen{};
    if (FAILED(g_resources.device1->CreateDeviceContextState(
            0, levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
            __uuidof(ID3D11Device), &chosen, &g_resources.overlayState))) return false;

    D3D11_TEXTURE2D_DESC copyDesc = description;
    copyDesc.Usage = D3D11_USAGE_DEFAULT;
    copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    copyDesc.CPUAccessFlags = 0;
    copyDesc.MiscFlags = 0;
    copyDesc.SampleDesc.Count = 1;
    copyDesc.SampleDesc.Quality = 0;
    if (FAILED(device->CreateTexture2D(&copyDesc, nullptr, &g_resources.copyTexture))) return false;
    if (FAILED(device->CreateShaderResourceView(
            g_resources.copyTexture.Get(), nullptr, &g_resources.sourceView))) return false;
    if (FAILED(device->CreateRenderTargetView(
            backBuffer, nullptr, &g_resources.backBufferView))) return false;

    ComPtr<ID3DBlob> vs;
    ComPtr<ID3DBlob> ps;
    if (FAILED(compile_shader(kVertexShader, "vs_5_0", vs)) ||
        FAILED(compile_shader(kPixelShader, "ps_5_0", ps))) return false;
    if (FAILED(device->CreateVertexShader(
            vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, &g_resources.vertexShader))) return false;
    if (FAILED(device->CreatePixelShader(
            ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, &g_resources.pixelShader))) return false;

    D3D11_BUFFER_DESC bufferDesc{};
    bufferDesc.ByteWidth = 64;
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device->CreateBuffer(&bufferDesc, nullptr, &g_resources.parameterBuffer))) return false;

    D3D11_SAMPLER_DESC samplerDesc{};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device->CreateSamplerState(&samplerDesc, &g_resources.sampler))) return false;

    D3D11_RASTERIZER_DESC rasterDesc{};
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    rasterDesc.DepthClipEnable = TRUE;
    if (FAILED(device->CreateRasterizerState(&rasterDesc, &g_resources.rasterizer))) return false;

    D3D11_BLEND_DESC blendDesc{};
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device->CreateBlendState(&blendDesc, &g_resources.blend))) return false;

    D3D11_DEPTH_STENCIL_DESC depthDesc{};
    depthDesc.DepthEnable = FALSE;
    depthDesc.StencilEnable = FALSE;
    if (FAILED(device->CreateDepthStencilState(&depthDesc, &g_resources.depthStencil))) return false;

    if (!make_d2d_target(backBuffer)) return false;
    g_resources.ready = true;
    return true;
}

void draw_text(
    ID2D1RenderTarget* target, IDWriteTextFormat* format,
    const std::wstring& value, const D2D1_RECT_F& bounds,
    ID2D1Brush* brush) {
    target->DrawTextW(
        value.c_str(), static_cast<UINT32>(value.size()), format,
        bounds, brush, D2D1_DRAW_TEXT_OPTIONS_CLIP,
        DWRITE_MEASURING_MODE_NATURAL);
}

float read_slider_value(const Settings& value, const SliderDefinition& slider) {
    return value.*(slider.member);
}

void update_slider_from_mouse(int x, int y) {
    if (g_dragSlider < 0 || g_dragSlider >= kSliderCount) return;
    const auto& slider = kSliders[static_cast<std::size_t>(g_dragSlider)];
    const float trackStart = kMenuLeft + static_cast<float>(kSliderTrackLeft);
    const float trackEnd = kMenuLeft + static_cast<float>(kSliderTrackRight);
    const float fraction = clamp_float((static_cast<float>(x) - trackStart) / (trackEnd - trackStart), 0.0f, 1.0f);
    {
        std::lock_guard<std::mutex> lock(g_settingsMutex);
        g_settings.*(slider.member) = slider.minimum + fraction * (slider.maximum - slider.minimum);
        g_settings.activePreset = L"Custom";
    }
    (void)y;
}

bool point_inside_menu(int x, int y) {
    return x >= static_cast<int>(kMenuLeft) && x <= static_cast<int>(kMenuLeft + kMenuWidth)
        && y >= static_cast<int>(kMenuTop) && y <= static_cast<int>(kMenuTop + kMenuHeight);
}

void handle_menu_click(HWND hwnd, int x, int y) {
    const int localX = x - static_cast<int>(kMenuLeft);
    const int localY = y - static_cast<int>(kMenuTop);
    if (localY >= 54 && localY <= 112) {
        const int column = (localX - 12) / 165;
        if (column >= 0 && column < 3) {
            if (localY <= 83) {
                const wchar_t* presets[] = {L"Neutral", L"Cinematic", L"Vibrant"};
                set_preset(presets[column]);
            } else {
                const wchar_t* presets[] = {L"Clarity", L"Night", L"Film"};
                set_preset(presets[column]);
            }
            return;
        }
    }
    const int saveY = static_cast<int>(kMenuHeight) - 48;
    if (localY >= saveY && localY <= saveY + 30) {
        if (localX >= 14 && localX <= 166) (void)save_preset_dialog(hwnd);
        else if (localX >= 172 && localX <= 324) (void)load_preset_dialog(hwnd);
        else if (localX >= 330 && localX <= 505) {
            {
                std::lock_guard<std::mutex> lock(g_settingsMutex);
                g_settings.enabled = !g_settings.enabled;
                g_enabled = g_settings.enabled;
            }
            save_settings_file();
        }
        return;
    }
    if (localY >= 15 && localY <= 43 && localX >= 380) {
        {
            std::lock_guard<std::mutex> lock(g_settingsMutex);
            g_settings.enabled = !g_settings.enabled;
            g_enabled = g_settings.enabled;
        }
        save_settings_file();
        return;
    }
    const int row = (localY - kSliderStartY) / kSliderRowHeight + g_scrollOffset;
    if (row >= 0 && row < kSliderCount &&
        localX >= kSliderTrackLeft - 12 && localX <= kSliderTrackRight + 12) {
        g_dragSlider = row;
        update_slider_from_mouse(x, y);
        SetCapture(hwnd);
    }
}

void draw_menu() {
    if (!g_menuVisible.load() || !g_resources.d2dTarget) return;
    ID2D1RenderTarget* target = g_resources.d2dTarget.Get();
    // Reuse brushes created with the back-buffer target to avoid per-frame COM allocations.
    ID2D1SolidColorBrush* white = g_resources.textBrush.Get();
    ID2D1SolidColorBrush* muted = g_resources.mutedBrush.Get();
    ID2D1SolidColorBrush* accent = g_resources.accentBrush.Get();
    ID2D1SolidColorBrush* panel = g_resources.panelBrush.Get();
    ID2D1SolidColorBrush* panelHeader = g_resources.panelHeaderBrush.Get();
    ID2D1SolidColorBrush* sliderBack = g_resources.sliderBackBrush.Get();
    ID2D1SolidColorBrush* button = g_resources.buttonBrush.Get();
    if (!white || !muted || !accent || !panel || !panelHeader || !sliderBack || !button) return;

    target->BeginDraw();
    const D2D1_RECT_F menu = D2D1::RectF(
        kMenuLeft, kMenuTop, kMenuLeft + kMenuWidth, kMenuTop + kMenuHeight);
    target->FillRoundedRectangle(D2D1::RoundedRect(menu, 9.0f, 9.0f), panel);
    target->FillRoundedRectangle(
        D2D1::RoundedRect(D2D1::RectF(kMenuLeft, kMenuTop, kMenuLeft + kMenuWidth, kMenuTop + 47.0f), 9.0f, 9.0f),
        panelHeader);
    draw_text(target, g_resources.titleFormat.Get(), L"TrSS  /  REAL-TIME LOOK",
        D2D1::RectF(kMenuLeft + 16.0f, kMenuTop + 11.0f, kMenuLeft + 365.0f, kMenuTop + 42.0f), white);

    Settings value;
    { std::lock_guard<std::mutex> lock(g_settingsMutex); value = g_settings; }
    const std::wstring toggle = value.enabled ? L"EFFECTS  ON" : L"EFFECTS  OFF";
    draw_text(target, g_resources.smallFormat.Get(), toggle,
        D2D1::RectF(kMenuLeft + 385.0f, kMenuTop + 16.0f, kMenuLeft + 505.0f, kMenuTop + 41.0f),
        value.enabled ? accent : muted);

    draw_text(target, g_resources.smallFormat.Get(), L"PRESETS  /  SELECT A LOOK",
        D2D1::RectF(kMenuLeft + 14.0f, kMenuTop + 48.0f, kMenuLeft + 260.0f, kMenuTop + 65.0f), muted);
    const wchar_t* presetLabels[] = {
        L"Neutral", L"Cinematic", L"Vibrant", L"Clarity", L"Night", L"Film"
    };
    for (int index = 0; index < 6; ++index) {
        const int column = index % 3;
        const int row = index / 3;
        const float left = kMenuLeft + 12.0f + static_cast<float>(column) * 165.0f;
        const float top = kMenuTop + 57.0f + static_cast<float>(row) * 29.0f;
        const auto isActive = _wcsicmp(value.activePreset.c_str(), presetLabels[index]) == 0;
        if (isActive) {
            target->FillRoundedRectangle(
                D2D1::RoundedRect(D2D1::RectF(left, top, left + 158.0f, top + 25.0f), 4.0f, 4.0f), accent);
            draw_text(target, g_resources.buttonFormat.Get(), presetLabels[index],
                D2D1::RectF(left + 5.0f, top + 2.0f, left + 153.0f, top + 23.0f), panel);
        } else {
            target->FillRoundedRectangle(
                D2D1::RoundedRect(D2D1::RectF(left, top, left + 158.0f, top + 25.0f), 4.0f, 4.0f), button);
            draw_text(target, g_resources.buttonFormat.Get(), presetLabels[index],
                D2D1::RectF(left + 5.0f, top + 2.0f, left + 153.0f, top + 23.0f), white);
        }
    }

    draw_text(target, g_resources.smallFormat.Get(),
        L"POST-PROCESSING  /  CHANGES PREVIEW ON THE LIVE FRAME",
        D2D1::RectF(kMenuLeft + 14.0f, kMenuTop + 115.0f, kMenuLeft + 500.0f, kMenuTop + 132.0f), muted);

    for (int index = 0; index < kSliderCount; ++index) {
        const auto& slider = kSliders[static_cast<std::size_t>(index)];
        const float top = kMenuTop + static_cast<float>(kSliderStartY + index * kSliderRowHeight - g_scrollOffset * kSliderRowHeight);
        if (top < kMenuTop + 129.0f || top > kMenuTop + kMenuHeight - 56.0f) continue;
        const float current = read_slider_value(value, slider);
        const float fraction = clamp_float((current - slider.minimum) / (slider.maximum - slider.minimum), 0.0f, 1.0f);
        draw_text(target, g_resources.textFormat.Get(), slider.label,
            D2D1::RectF(kMenuLeft + 16.0f, top + 2.0f, kMenuLeft + 210.0f, top + 25.0f), white);
        const float trackLeft = kMenuLeft + static_cast<float>(kSliderTrackLeft);
        const float trackRight = kMenuLeft + static_cast<float>(kSliderTrackRight);
        target->FillRoundedRectangle(
            D2D1::RoundedRect(D2D1::RectF(trackLeft, top + 10.0f, trackRight, top + 19.0f), 3.0f, 3.0f),
            sliderBack);
        target->FillRoundedRectangle(
            D2D1::RoundedRect(D2D1::RectF(trackLeft, top + 10.0f, trackLeft + (trackRight - trackLeft) * fraction, top + 19.0f), 3.0f, 3.0f),
            accent);
        target->FillEllipse(D2D1::Ellipse(
            D2D1::Point2F(trackLeft + (trackRight - trackLeft) * fraction, top + 14.5f), 5.3f, 5.3f), white);
        wchar_t number[40]{};
        swprintf_s(number, slider.format, static_cast<double>(current));
        draw_text(target, g_resources.smallFormat.Get(), number,
            D2D1::RectF(kMenuLeft + 427.0f, top + 1.0f, kMenuLeft + 506.0f, top + 25.0f), muted);
    }

    const float footerY = kMenuTop + kMenuHeight - 48.0f;
    draw_text(target, g_resources.smallFormat.Get(), L"Ctrl + Alt + Page Up  /  close menu",
        D2D1::RectF(kMenuLeft + 15.0f, footerY - 20.0f, kMenuLeft + 490.0f, footerY - 3.0f), muted);
    const wchar_t* buttons[] = {L"Save preset", L"Load preset", value.enabled ? L"Disable effects" : L"Enable effects"};
    const float widths[] = {152.0f, 152.0f, 174.0f};
    float buttonX = kMenuLeft + 12.0f;
    for (int index = 0; index < 3; ++index) {
        target->FillRoundedRectangle(
            D2D1::RoundedRect(D2D1::RectF(buttonX, footerY, buttonX + widths[index] - 6.0f, footerY + 30.0f), 4.0f, 4.0f),
            button);
        draw_text(target, g_resources.buttonFormat.Get(), buttons[index],
            D2D1::RectF(buttonX + 5.0f, footerY + 3.0f, buttonX + widths[index] - 11.0f, footerY + 27.0f), white);
        buttonX += widths[index];
    }
    draw_text(target, g_resources.smallFormat.Get(),
        L"First target: Direct3D 11 / DXGI. Neural upscaling needs compatible weights and real depth/motion.",
        D2D1::RectF(kMenuLeft + 14.0f, kMenuTop + kMenuHeight - 15.0f, kMenuLeft + 505.0f, kMenuTop + kMenuHeight + 5.0f),
        muted);
    target->EndDraw();
}

void hook_game_window(HWND window);

void render_frame(IDXGISwapChain* swapChain) {
    std::lock_guard<std::mutex> renderLock(g_renderMutex);
    ComPtr<ID3D11Device> device;
    if (FAILED(swapChain->GetDevice(IID_PPV_ARGS(&device)))) return;
    ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(&context);
    if (!context) return;
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return;
    D3D11_TEXTURE2D_DESC description{};
    backBuffer->GetDesc(&description);

    if (description.SampleDesc.Count != 1 ||
        (description.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
         description.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB &&
         description.Format != DXGI_FORMAT_B8G8R8A8_UNORM &&
         description.Format != DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)) {
        return;
    }

    if (!g_resources.ready || g_resources.device.Get() != device.Get() ||
        g_resources.width != description.Width || g_resources.height != description.Height ||
        g_resources.format != description.Format || g_resources.swapChain != swapChain) {
        reset_render_resources();
        if (!initialize_render_resources(swapChain, device.Get(), context.Get(), backBuffer.Get(), description)) {
            reset_render_resources();
            return;
        }
    }

    DXGI_SWAP_CHAIN_DESC swapDescription{};
    if (SUCCEEDED(swapChain->GetDesc(&swapDescription)) && swapDescription.OutputWindow) {
        hook_game_window(swapDescription.OutputWindow);
    }
    Settings value;
    { std::lock_guard<std::mutex> lock(g_settingsMutex); value = g_settings; }
    ComPtr<ID3DDeviceContextState> previousState;
    g_resources.context1->SwapDeviceContextState(
        g_resources.overlayState.Get(), &previousState);
    if (!previousState) return;

    // Save the original frame only after the game's render-target state is isolated.
    if (value.enabled) {
        g_resources.context->CopyResource(g_resources.copyTexture.Get(), backBuffer.Get());
    }

    if (value.enabled) {
        struct alignas(16) ShaderParameters {
            float values[16];
        } parameters{};
        parameters.values[0] = value.brightness;
        parameters.values[1] = value.contrast;
        parameters.values[2] = value.saturation;
        parameters.values[3] = value.gamma;
        parameters.values[4] = value.sharpen;
        parameters.values[5] = value.bloom;
        parameters.values[6] = value.vignette;
        parameters.values[7] = value.grain;
        parameters.values[8] = value.temperature;
        parameters.values[9] = value.chromatic;
        parameters.values[10] = g_animationTime += 0.016f;
        parameters.values[11] = 1.0f;
        parameters.values[12] = static_cast<float>(description.Width);
        parameters.values[13] = static_cast<float>(description.Height);
        g_resources.context->UpdateSubresource(g_resources.parameterBuffer.Get(), 0, nullptr, &parameters, 0, 0);

        D3D11_VIEWPORT viewport{};
        viewport.Width = static_cast<float>(description.Width);
        viewport.Height = static_cast<float>(description.Height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        g_resources.context->RSSetViewports(1, &viewport);
        g_resources.context->RSSetState(g_resources.rasterizer.Get());
        ID3D11RenderTargetView* renderTarget = g_resources.backBufferView.Get();
        g_resources.context->OMSetRenderTargets(1, &renderTarget, nullptr);
        g_resources.context->OMSetBlendState(g_resources.blend.Get(), nullptr, 0xFFFFFFFF);
        g_resources.context->OMSetDepthStencilState(g_resources.depthStencil.Get(), 0);
        g_resources.context->IASetInputLayout(nullptr);
        g_resources.context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_resources.context->VSSetShader(g_resources.vertexShader.Get(), nullptr, 0);
        g_resources.context->PSSetShader(g_resources.pixelShader.Get(), nullptr, 0);
        ID3D11ShaderResourceView* sourceView = g_resources.sourceView.Get();
        g_resources.context->PSSetShaderResources(0, 1, &sourceView);
        ID3D11SamplerState* sampler = g_resources.sampler.Get();
        g_resources.context->PSSetSamplers(0, 1, &sampler);
        ID3D11Buffer* constants = g_resources.parameterBuffer.Get();
        g_resources.context->PSSetConstantBuffers(0, 1, &constants);
        g_resources.context->Draw(3, 0);
        ID3D11ShaderResourceView* nullView = nullptr;
        g_resources.context->PSSetShaderResources(0, 1, &nullView);
    }
    draw_menu();
    g_resources.context1->SwapDeviceContextState(previousState.Get(), nullptr);
}

void reset_for_resize(IDXGISwapChain* swapChain) {
    std::lock_guard<std::mutex> lock(g_renderMutex);
    if (g_resources.swapChain == swapChain) reset_render_resources();
}

void patch_entry(void* instance, std::size_t index, void* hook, void** original) {
    if (!instance) return;
    auto** table = *reinterpret_cast<void***>(instance);
    std::lock_guard<std::mutex> lock(g_patchMutex);
    if (table[index] == hook) return;
    DWORD previousProtection = 0;
    if (!VirtualProtect(&table[index], sizeof(void*), PAGE_EXECUTE_READWRITE, &previousProtection)) return;
    if (original && *original == nullptr) *original = table[index];
    table[index] = hook;
    DWORD ignored = 0;
    VirtualProtect(&table[index], sizeof(void*), previousProtection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), &table[index], sizeof(void*));
}

HRESULT STDMETHODCALLTYPE hook_present(IDXGISwapChain* swapChain, UINT syncInterval, UINT flags) {
    if (g_realPresent == nullptr) return DXGI_ERROR_INVALID_CALL;
    bool expected = false;
    if (g_insidePresent.compare_exchange_strong(expected, true)) {
        __try {
            render_frame(swapChain);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            // Fail closed: if a graphics driver rejects a target or device state,
            // let the game continue presenting its own frame untouched.
        }
        g_insidePresent = false;
    }
    return g_realPresent(swapChain, syncInterval, flags);
}

HRESULT STDMETHODCALLTYPE hook_resize_buffers(
    IDXGISwapChain* swapChain, UINT bufferCount, UINT width, UINT height,
    DXGI_FORMAT format, UINT flags) {
    reset_for_resize(swapChain);
    return g_realResizeBuffers
        ? g_realResizeBuffers(swapChain, bufferCount, width, height, format, flags)
        : DXGI_ERROR_INVALID_CALL;
}

void patch_swap_chain(IDXGISwapChain* swapChain) {
    if (!swapChain) return;
    void** table = *reinterpret_cast<void***>(swapChain);
    patch_entry(swapChain, 8, reinterpret_cast<void*>(&hook_present), reinterpret_cast<void**>(&g_realPresent));
    patch_entry(swapChain, 13, reinterpret_cast<void*>(&hook_resize_buffers), reinterpret_cast<void**>(&g_realResizeBuffers));
    DXGI_SWAP_CHAIN_DESC desc{};
    if (SUCCEEDED(swapChain->GetDesc(&desc)) && desc.OutputWindow) hook_game_window(desc.OutputWindow);
    (void)table;
}

HRESULT STDMETHODCALLTYPE hook_create_swap_chain(
    IDXGIFactory* factory, IUnknown* device, DXGI_SWAP_CHAIN_DESC* description,
    IDXGISwapChain** swapChain) {
    if (!g_realCreateSwapChain) return DXGI_ERROR_INVALID_CALL;
    const HRESULT hr = g_realCreateSwapChain(factory, device, description, swapChain);
    if (SUCCEEDED(hr) && swapChain && *swapChain) patch_swap_chain(*swapChain);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_create_swap_chain_for_hwnd(
    IDXGIFactory2* factory, IUnknown* device, HWND window,
    const DXGI_SWAP_CHAIN_DESC1* description,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreenDescription,
    IDXGIOutput* restrictOutput, IDXGISwapChain1** swapChain) {
    if (!g_realCreateSwapChainForHwnd) return DXGI_ERROR_INVALID_CALL;
    const HRESULT hr = g_realCreateSwapChainForHwnd(
        factory, device, window, description, fullscreenDescription, restrictOutput, swapChain);
    if (SUCCEEDED(hr) && swapChain && *swapChain) {
        patch_swap_chain(*swapChain);
        hook_game_window(window);
    }
    return hr;
}

void patch_factory(void* factoryInterface, REFIID requestedIid) {
    if (!factoryInterface) return;
    // IDXGIFactory::CreateSwapChain is vtable slot 10 after IUnknown and IDXGIObject.
    patch_entry(factoryInterface, 10, reinterpret_cast<void*>(&hook_create_swap_chain),
        reinterpret_cast<void**>(&g_realCreateSwapChain));

    IUnknown* unknown = static_cast<IUnknown*>(factoryInterface);
    ComPtr<IDXGIFactory2> factory2;
    if (SUCCEEDED(unknown->QueryInterface(IID_PPV_ARGS(&factory2))) && factory2) {
        // IDXGIFactory2::CreateSwapChainForHwnd is slot 15.
        patch_entry(factory2.Get(), 15, reinterpret_cast<void*>(&hook_create_swap_chain_for_hwnd),
            reinterpret_cast<void**>(&g_realCreateSwapChainForHwnd));
    }
    (void)requestedIid;
}

void ensure_real_dxgi() {
    std::call_once(g_dxgiOnce, [] {
        wchar_t systemDirectory[MAX_PATH]{};
        const UINT length = GetSystemDirectoryW(systemDirectory, MAX_PATH);
        if (length == 0 || length >= MAX_PATH) return;
        std::wstring path(systemDirectory, length);
        path = join_path(path, L"dxgi.dll");
        g_realDxgi = LoadLibraryW(path.c_str());
        if (!g_realDxgi) return;
        g_realCreateFactory = reinterpret_cast<CreateFactoryFn>(
            GetProcAddress(g_realDxgi, "CreateDXGIFactory"));
        g_realCreateFactory1 = reinterpret_cast<CreateFactoryFn>(
            GetProcAddress(g_realDxgi, "CreateDXGIFactory1"));
        g_realCreateFactory2 = reinterpret_cast<CreateFactory2Fn>(
            GetProcAddress(g_realDxgi, "CreateDXGIFactory2"));
        g_realDeclareRemovalSupport = reinterpret_cast<DeclareRemovalFn>(
            GetProcAddress(g_realDxgi, "DXGIDeclareAdapterRemovalSupport"));
        g_realDisableVBlankVirtualization = reinterpret_cast<DisableVBlankFn>(
            GetProcAddress(g_realDxgi, "DXGIDisableVBlankVirtualization"));
        g_realGetDebugInterface = reinterpret_cast<GetDebugInterfaceFn>(
            GetProcAddress(g_realDxgi, "DXGIGetDebugInterface"));
        g_realGetDebugInterface1 = reinterpret_cast<GetDebugInterface1Fn>(
            GetProcAddress(g_realDxgi, "DXGIGetDebugInterface1"));
    });
}

HRESULT finish_factory_call(HRESULT hr, REFIID iid, void** factory) {
    if (SUCCEEDED(hr) && factory && *factory) patch_factory(*factory, iid);
    return hr;
}

LRESULT CALLBACK runtime_window_proc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    const bool visible = g_menuVisible.load();
    if (message == WM_KEYDOWN && wParam == VK_PRIOR &&
        (GetKeyState(VK_CONTROL) & 0x8000) && (GetKeyState(VK_MENU) & 0x8000)) {
        g_menuVisible = !g_menuVisible.load();
        g_dragSlider = -1;
        return 0;
    }
    if (visible && message == WM_MOUSEWHEEL) {
        const short delta = GET_WHEEL_DELTA_WPARAM(wParam);
        g_scrollOffset = std::max(0, std::min(kSliderCount - 6, g_scrollOffset + (delta < 0 ? 1 : -1)));
        return 0;
    }
    if (visible && (message == WM_LBUTTONDOWN || message == WM_LBUTTONUP ||
                    message == WM_MOUSEMOVE)) {
        const int x = GET_X_LPARAM(lParam);
        const int y = GET_Y_LPARAM(lParam);
        if (point_inside_menu(x, y) || g_dragSlider >= 0) {
            if (message == WM_LBUTTONDOWN) {
                handle_menu_click(hwnd, x, y);
                return 0;
            }
            if (message == WM_MOUSEMOVE && g_dragSlider >= 0 &&
                (wParam & MK_LBUTTON) != 0) {
                update_slider_from_mouse(x, y);
                return 0;
            }
            if (message == WM_LBUTTONUP) {
                if (g_dragSlider >= 0) {
                    update_slider_from_mouse(x, y);
                    g_dragSlider = -1;
                    ReleaseCapture();
                    save_settings_file();
                }
                return 0;
            }
            if (message == WM_MOUSEMOVE && point_inside_menu(x, y)) return 0;
        }
    }
    return g_previousWndProc ? CallWindowProcW(g_previousWndProc, hwnd, message, wParam, lParam)
                             : DefWindowProcW(hwnd, message, wParam, lParam);
}

void hook_game_window(HWND window) {
    if (!window || !IsWindow(window)) return;
    if (g_gameWindow == window && g_previousWndProc) return;
    if (g_gameWindow && g_previousWndProc && IsWindow(g_gameWindow)) {
        SetWindowLongPtrW(g_gameWindow, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_previousWndProc));
    }
    SetLastError(0);
    const LONG_PTR oldProc = SetWindowLongPtrW(
        window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&runtime_window_proc));
    if (oldProc == 0 && GetLastError() != 0) return;
    g_gameWindow = window;
    g_previousWndProc = reinterpret_cast<WNDPROC>(oldProc);
}

} // namespace trss_runtime

extern "C" HRESULT WINAPI TrSSCreateDXGIFactory(REFIID iid, void** factory) {
    using namespace trss_runtime;
    load_settings_file();
    ensure_real_dxgi();
    if (!g_realCreateFactory) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    return finish_factory_call(g_realCreateFactory(iid, factory), iid, factory);
}

extern "C" HRESULT WINAPI TrSSCreateDXGIFactory1(REFIID iid, void** factory) {
    using namespace trss_runtime;
    load_settings_file();
    ensure_real_dxgi();
    if (!g_realCreateFactory1) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    return finish_factory_call(g_realCreateFactory1(iid, factory), iid, factory);
}

extern "C" HRESULT WINAPI TrSSCreateDXGIFactory2(UINT flags, REFIID iid, void** factory) {
    using namespace trss_runtime;
    load_settings_file();
    ensure_real_dxgi();
    if (!g_realCreateFactory2) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    return finish_factory_call(g_realCreateFactory2(flags, iid, factory), iid, factory);
}

extern "C" HRESULT WINAPI TrSSDXGIDeclareAdapterRemovalSupport() {
    using namespace trss_runtime;
    ensure_real_dxgi();
    if (!g_realDeclareRemovalSupport) return S_OK;
    return g_realDeclareRemovalSupport();
}

extern "C" HRESULT WINAPI TrSSDXGIDisableVBlankVirtualization() {
    using namespace trss_runtime;
    ensure_real_dxgi();
    if (!g_realDisableVBlankVirtualization) return E_NOTIMPL;
    return g_realDisableVBlankVirtualization();
}

extern "C" HRESULT WINAPI TrSSDXGIGetDebugInterface(REFIID iid, void** debugInterface) {
    using namespace trss_runtime;
    ensure_real_dxgi();
    if (!g_realGetDebugInterface) return DXGI_ERROR_NOT_FOUND;
    return g_realGetDebugInterface(iid, debugInterface);
}

extern "C" HRESULT WINAPI TrSSDXGIGetDebugInterface1(
    UINT flags, REFIID iid, void** debugInterface) {
    using namespace trss_runtime;
    ensure_real_dxgi();
    if (!g_realGetDebugInterface1) return DXGI_ERROR_NOT_FOUND;
    return g_realGetDebugInterface1(flags, iid, debugInterface);
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
    using namespace trss_runtime;
    if (reason == DLL_PROCESS_ATTACH) {
        // Keep DllMain minimal. Path discovery and heap-backed strings happen only
        // after Windows has finished loading this module, at the first factory call.
        g_module = module;
        DisableThreadLibraryCalls(module);
    } else if (reason == DLL_PROCESS_DETACH) {
        // Do not call Direct3D, window, or loader APIs from DllMain during process
        // detach: Windows may already hold the loader lock and the process is exiting.
    }
    return TRUE;
}
