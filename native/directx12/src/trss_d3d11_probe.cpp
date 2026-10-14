// TrSS Direct3D 11 compute bring-up for legacy-game API coverage.
//
// This executable validates adapter selection, DirectCompute, UAV writes, and
// CPU readback. It does not hook a game or execute TrSS neural model weights.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT kElementCount = 1024;
constexpr UINT kThreadsPerGroup = 64;

struct AdapterInfo {
    ComPtr<IDXGIAdapter1> adapter;
    std::string name;
    UINT vendor_id = 0;
    UINT device_id = 0;
    bool software = false;
    bool available = false;
    D3D_FEATURE_LEVEL level = static_cast<D3D_FEATURE_LEVEL>(0);
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

std::string wide_to_utf8(const wchar_t* value) {
    if (value == nullptr || value[0] == L'\0') return {};
    const int needed = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) return {};
    std::string text(static_cast<std::size_t>(needed), '\0');
    const int written = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, text.data(), needed, nullptr, nullptr);
    if (written <= 0) return {};
    text.pop_back();
    return text;
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

std::string feature_level_name(D3D_FEATURE_LEVEL level) {
    switch (level) {
    case D3D_FEATURE_LEVEL_12_1: return "12_1";
    case D3D_FEATURE_LEVEL_12_0: return "12_0";
    case D3D_FEATURE_LEVEL_11_1: return "11_1";
    case D3D_FEATURE_LEVEL_11_0: return "11_0";
    case D3D_FEATURE_LEVEL_10_1: return "10_1";
    case D3D_FEATURE_LEVEL_10_0: return "10_0";
    case D3D_FEATURE_LEVEL_9_3: return "9_3";
    case D3D_FEATURE_LEVEL_9_2: return "9_2";
    case D3D_FEATURE_LEVEL_9_1: return "9_1";
    default: return "not_available";
    }
}

HRESULT create_device(
    IDXGIAdapter* adapter,
    D3D_DRIVER_TYPE driver_type,
    ComPtr<ID3D11Device>& device,
    ComPtr<ID3D11DeviceContext>& context,
    D3D_FEATURE_LEVEL* selected_level) {
    const D3D_FEATURE_LEVEL levels11_1[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3, D3D_FEATURE_LEVEL_9_2, D3D_FEATURE_LEVEL_9_1
    };
    HRESULT result = D3D11CreateDevice(
        adapter, driver_type, nullptr, 0, levels11_1,
        static_cast<UINT>(std::size(levels11_1)), D3D11_SDK_VERSION,
        &device, selected_level, &context);
    if (result == E_INVALIDARG) {
        const D3D_FEATURE_LEVEL levels10[] = {
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3, D3D_FEATURE_LEVEL_9_2, D3D_FEATURE_LEVEL_9_1
        };
        result = D3D11CreateDevice(
            adapter, driver_type, nullptr, 0, levels10,
            static_cast<UINT>(std::size(levels10)), D3D11_SDK_VERSION,
            &device, selected_level, &context);
    }
    return result;
}

AdapterInfo describe_adapter(ComPtr<IDXGIAdapter1> adapter) {
    DXGI_ADAPTER_DESC1 desc{};
    check_hresult(adapter->GetDesc1(&desc), "IDXGIAdapter1::GetDesc1");
    AdapterInfo info;
    info.adapter = std::move(adapter);
    info.name = wide_to_utf8(desc.Description);
    info.vendor_id = desc.VendorId;
    info.device_id = desc.DeviceId;
    info.software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    const HRESULT result = create_device(
        info.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, device, context, &info.level);
    info.available = SUCCEEDED(result);
    if (!info.available) {
        std::ostringstream reason;
        reason << "D3D11CreateDevice failed (HRESULT 0x"
               << std::hex << std::uppercase
               << static_cast<unsigned long>(static_cast<std::uint32_t>(result)) << ")";
        info.reason = reason.str();
        info.level = static_cast<D3D_FEATURE_LEVEL>(0);
    }
    return info;
}

std::vector<AdapterInfo> enumerate_adapters() {
    ComPtr<IDXGIFactory1> factory;
    check_hresult(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    std::vector<AdapterInfo> adapters;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "IDXGIFactory1::EnumAdapters1");
        adapters.push_back(describe_adapter(std::move(adapter)));
    }
    return adapters;
}

void print_devices(const std::vector<AdapterInfo>& adapters) {
    std::cout << "{\n  \"api\": \"Direct3D 11\",\n  \"devices\": [";
    if (!adapters.empty()) std::cout << '\n';
    for (std::size_t i = 0; i < adapters.size(); ++i) {
        const auto& item = adapters[i];
        std::cout << "    {\n"
                  << "      \"name\": " << quote_json(item.name) << ",\n"
                  << "      \"vendor_id\": " << quote_json(hex_id(item.vendor_id)) << ",\n"
                  << "      \"device_id\": " << quote_json(hex_id(item.device_id)) << ",\n"
                  << "      \"software_adapter\": " << (item.software ? "true" : "false") << ",\n"
                  << "      \"device_created\": " << (item.available ? "true" : "false") << ",\n"
                  << "      \"feature_level\": " << quote_json(feature_level_name(item.level));
        if (!item.reason.empty()) std::cout << ",\n      \"reason\": " << quote_json(item.reason);
        std::cout << "\n    }" << (i + 1 == adapters.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n"
              << "  \"note\": \"Device discovery is not a game-hook or neural-inference test.\"\n}\n";
}

std::size_t select_adapter(
    const std::vector<AdapterInfo>& adapters, const std::string& requested) {
    const std::string needle = lowercase(requested);
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < adapters.size(); ++i) {
        const auto& item = adapters[i];
        if (!item.available || item.software || item.level < D3D_FEATURE_LEVEL_11_0) continue;
        const auto name = lowercase(item.name);
        if ((!needle.empty() && name.find(needle) != std::string::npos)
            || (needle.empty() && name.find("arc") != std::string::npos)) {
            matches.push_back(i);
        }
    }
    if (matches.size() == 1) return matches.front();
    if (matches.empty()) {
        throw std::runtime_error(
            "No unique Direct3D 11 compute-capable Arc adapter matched. "
            "Use --list-devices and optionally specify --device-substring.");
    }
    throw std::runtime_error("Adapter name matched multiple devices; provide a longer substring.");
}

std::string blob_text(ID3DBlob* blob) {
    if (blob == nullptr || blob->GetBufferPointer() == nullptr) return {};
    return std::string(
        static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
}

void execute_compute_roundtrip(ID3D11Device* device, ID3D11DeviceContext* context) {
    const UINT byte_count = kElementCount * static_cast<UINT>(sizeof(std::uint32_t));
    const char* source = R"(
        RWByteAddressBuffer OutputData : register(u0);
        cbuffer Parameters : register(b0) { uint ElementCount; uint3 Padding; };
        [numthreads(64, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint index = id.x;
            if (index < ElementCount) OutputData.Store(index * 4, index * 3 + 7);
        }
    )";
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT result = D3DCompile(
        source, std::char_traits<char>::length(source), "trss_d3d11_test.hlsl",
        nullptr, nullptr, "main", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0, &shader, &errors);
    if (FAILED(result)) throw std::runtime_error("D3D11 HLSL compile failed: " + blob_text(errors.Get()));

    ComPtr<ID3D11ComputeShader> compute;
    check_hresult(device->CreateComputeShader(
        shader->GetBufferPointer(), shader->GetBufferSize(), nullptr, &compute),
        "ID3D11Device::CreateComputeShader");

    D3D11_BUFFER_DESC output_desc{};
    output_desc.ByteWidth = byte_count;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    output_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    ComPtr<ID3D11Buffer> output;
    check_hresult(device->CreateBuffer(&output_desc, nullptr, &output),
        "ID3D11Device::CreateBuffer (output)");

    D3D11_UNORDERED_ACCESS_VIEW_DESC view_desc{};
    view_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    view_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    view_desc.Buffer.NumElements = kElementCount;
    view_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    ComPtr<ID3D11UnorderedAccessView> view;
    check_hresult(device->CreateUnorderedAccessView(output.Get(), &view_desc, &view),
        "ID3D11Device::CreateUnorderedAccessView");

    UINT constants[4] = {kElementCount, 0, 0, 0};
    D3D11_BUFFER_DESC constants_desc{};
    constants_desc.ByteWidth = sizeof(constants);
    constants_desc.Usage = D3D11_USAGE_DEFAULT;
    constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constants_data{};
    constants_data.pSysMem = constants;
    ComPtr<ID3D11Buffer> constant_buffer;
    check_hresult(device->CreateBuffer(&constants_desc, &constants_data, &constant_buffer),
        "ID3D11Device::CreateBuffer (constant data)");

    D3D11_BUFFER_DESC read_desc{};
    read_desc.ByteWidth = byte_count;
    read_desc.Usage = D3D11_USAGE_STAGING;
    read_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> readback;
    check_hresult(device->CreateBuffer(&read_desc, nullptr, &readback),
        "ID3D11Device::CreateBuffer (readback)");

    context->CSSetShader(compute.Get(), nullptr, 0);
    ID3D11UnorderedAccessView* views[] = {view.Get()};
    context->CSSetUnorderedAccessViews(0, 1, views, nullptr);
    ID3D11Buffer* buffers[] = {constant_buffer.Get()};
    context->CSSetConstantBuffers(0, 1, buffers);
    context->Dispatch((kElementCount + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);
    ID3D11UnorderedAccessView* null_view = nullptr;
    context->CSSetUnorderedAccessViews(0, 1, &null_view, nullptr);
    context->CSSetShader(nullptr, nullptr, 0);
    context->CopyResource(readback.Get(), output.Get());
    context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    check_hresult(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped),
        "ID3D11DeviceContext::Map");
    const auto* values = static_cast<const std::uint32_t*>(mapped.pData);
    for (UINT i = 0; i < kElementCount; ++i) {
        const std::uint32_t expected = i * 3U + 7U;
        if (values[i] != expected) {
            context->Unmap(readback.Get(), 0);
            throw std::runtime_error("D3D11 compute output mismatch at element " + std::to_string(i));
        }
    }
    context->Unmap(readback.Get(), 0);
}

void run_warp_test() {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> warp;
    check_hresult(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    check_hresult(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "EnumWarpAdapter");
    AdapterInfo adapter = describe_adapter(std::move(warp));
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level = static_cast<D3D_FEATURE_LEVEL>(0);
    check_hresult(create_device(
        adapter.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, device, context, &level),
        "D3D11CreateDevice (WARP)");
    if (level < D3D_FEATURE_LEVEL_11_0) {
        throw std::runtime_error("WARP did not expose feature level 11_0 for compute.");
    }
    execute_compute_roundtrip(device.Get(), context.Get());
    std::cout << "{\n  \"api\": \"Direct3D 11\",\n"
              << "  \"backend\": \"Direct3D 11 compute (WARP software adapter)\",\n"
              << "  \"status\": \"passed\",\n"
              << "  \"adapter\": " << quote_json(adapter.name) << ",\n"
              << "  \"feature_level\": " << quote_json(feature_level_name(level)) << ",\n"
              << "  \"test\": \"hlsl_compute_uav_readback\",\n"
              << "  \"verified_elements\": " << kElementCount << ",\n"
              << "  \"note\": \"This does not test Intel hardware, trained neural inference, or game capture.\"\n}\n";
}

void usage() {
    std::cout << "TrSS Direct3D 11 API probe\n"
                 "  trss_d3d11_probe.exe --list-devices\n"
                 "  trss_d3d11_probe.exe --self-test [--device-substring NAME]\n"
                 "  trss_d3d11_probe.exe --warp-self-test\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool list = false;
        bool self_test = false;
        bool warp = false;
        std::string needle;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--list-devices") { list = true; continue; }
            if (arg == "--self-test") { self_test = true; continue; }
            if (arg == "--warp-self-test") { warp = true; continue; }
            if (arg == "--device-substring" && i + 1 < argc) {
                needle = argv[++i];
                if (needle.empty()) throw std::runtime_error("--device-substring cannot be blank.");
                continue;
            }
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (static_cast<int>(list) + static_cast<int>(self_test) + static_cast<int>(warp) > 1) {
            throw std::runtime_error("Choose one operation.");
        }
        if (!needle.empty() && (list || warp)) {
            throw std::runtime_error("--device-substring is only valid with --self-test.");
        }
        if (list) { print_devices(enumerate_adapters()); return 0; }
        if (warp) { run_warp_test(); return 0; }

        const auto adapters = enumerate_adapters();
        const auto selected = select_adapter(adapters, needle);
        const auto& adapter = adapters[selected];
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL level = static_cast<D3D_FEATURE_LEVEL>(0);
        check_hresult(create_device(
            adapter.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, device, context, &level),
            "D3D11CreateDevice (hardware)");
        execute_compute_roundtrip(device.Get(), context.Get());
        std::cout << "{\n  \"api\": \"Direct3D 11\",\n"
                  << "  \"backend\": \"Direct3D 11 compute (hardware adapter)\",\n"
                  << "  \"status\": \"passed\",\n"
                  << "  \"adapter\": " << quote_json(adapter.name) << ",\n"
                  << "  \"vendor_id\": " << quote_json(hex_id(adapter.vendor_id)) << ",\n"
                  << "  \"device_id\": " << quote_json(hex_id(adapter.device_id)) << ",\n"
                  << "  \"feature_level\": " << quote_json(feature_level_name(level)) << ",\n"
                  << "  \"test\": \"hlsl_compute_uav_readback\",\n"
                  << "  \"verified_elements\": " << kElementCount << ",\n"
                  << "  \"note\": \"This does not test the TrSS neural model or game capture.\"\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TrSS Direct3D 11 error: " << error.what() << '\n';
        return 2;
    }
}
