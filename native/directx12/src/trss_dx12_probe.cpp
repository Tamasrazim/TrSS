// TrSS Direct3D 12 bring-up backend.
//
// This executable proves that a selected D3D12 adapter can create a compute
// pipeline, write a GPU UAV buffer, synchronize, and return verified results.
// It deliberately does not claim to run the trained TrSS neural model yet.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

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

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT kElementCount = 1024;
constexpr UINT kThreadsPerGroup = 64;
// Keep a broken driver from leaving this diagnostic hung forever.
constexpr DWORD kFenceWaitTimeoutMs = 30000;

struct AdapterInfo {
    ComPtr<IDXGIAdapter1> adapter;
    std::string name;
    UINT vendor_id = 0;
    UINT device_id = 0;
    bool software = false;
    bool supports_d3d12 = false;
    std::string device_error;
};

class ScopedHandle {
public:
    explicit ScopedHandle(HANDLE handle) noexcept : handle_(handle) {}
    ~ScopedHandle() {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
    }

    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;

    HANDLE get() const noexcept { return handle_; }

private:
    HANDLE handle_ = nullptr;
};

std::string hresult_text(HRESULT result) {
    std::ostringstream formatted;
    formatted << "HRESULT 0x"
              << std::hex << std::uppercase
              << static_cast<unsigned long>(static_cast<std::uint32_t>(result));
    return formatted.str();
}

void check_hresult(HRESULT result, const char* operation) {
    if (SUCCEEDED(result)) {
        return;
    }
    throw std::runtime_error(std::string(operation) + " failed (" + hresult_text(result) + ").");
}

std::string wide_to_utf8(const wchar_t* value) {
    if (value == nullptr || value[0] == L'\0') {
        return {};
    }
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) {
        return {};
    }
    std::string result(static_cast<std::size_t>(required), '\0');
    const int written = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr);
    if (written <= 0) {
        return {};
    }
    result.pop_back();
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (character < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned int>(character)
                        << std::dec << std::setfill(' ');
            } else {
                escaped << static_cast<char>(character);
            }
        }
    }
    return escaped.str();
}

std::string quote_json(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string hex_id(UINT value) {
    std::ostringstream formatted;
    formatted << "0x" << std::uppercase << std::hex
              << std::setw(4) << std::setfill('0') << value;
    return formatted.str();
}

std::string lowercase(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

ComPtr<IDXGIFactory4> create_factory() {
    ComPtr<IDXGIFactory4> factory;
    check_hresult(
        CreateDXGIFactory1(IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory1");
    return factory;
}

AdapterInfo describe_adapter(ComPtr<IDXGIAdapter1> adapter) {
    DXGI_ADAPTER_DESC1 description{};
    check_hresult(adapter->GetDesc1(&description), "IDXGIAdapter1::GetDesc1");

    AdapterInfo info;
    info.adapter = std::move(adapter);
    info.name = wide_to_utf8(description.Description);
    info.vendor_id = description.VendorId;
    info.device_id = description.DeviceId;
    info.software = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

    ComPtr<ID3D12Device> probe_device;
    const HRESULT result = D3D12CreateDevice(
        info.adapter.Get(),
        D3D_FEATURE_LEVEL_12_0,
        IID_PPV_ARGS(&probe_device));
    info.supports_d3d12 = SUCCEEDED(result);
    if (FAILED(result)) {
        std::ostringstream message;
        message << "D3D12 feature level 12_0 unavailable (HRESULT 0x"
                << std::hex << std::uppercase
                << static_cast<unsigned long>(static_cast<std::uint32_t>(result))
                << ")";
        info.device_error = message.str();
    }
    return info;
}

std::vector<AdapterInfo> enumerate_adapters() {
    const auto factory = create_factory();
    std::vector<AdapterInfo> adapters;

    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        check_hresult(result, "IDXGIFactory1::EnumAdapters1");
        adapters.push_back(describe_adapter(std::move(adapter)));
    }
    return adapters;
}

void print_adapter_list(const std::vector<AdapterInfo>& adapters) {
    std::cout << "{\n  \"backend\": \"Direct3D 12\",\n  \"devices\": [";
    if (!adapters.empty()) {
        std::cout << '\n';
    }

    for (std::size_t index = 0; index < adapters.size(); ++index) {
        const auto& adapter = adapters[index];
        std::cout << "    {\n"
                  << "      \"name\": " << quote_json(adapter.name) << ",\n"
                  << "      \"vendor_id\": " << quote_json(hex_id(adapter.vendor_id)) << ",\n"
                  << "      \"device_id\": " << quote_json(hex_id(adapter.device_id)) << ",\n"
                  << "      \"software_adapter\": "
                  << (adapter.software ? "true" : "false") << ",\n"
                  << "      \"d3d12_feature_level_12_0\": "
                  << (adapter.supports_d3d12 ? "true" : "false");

        if (!adapter.device_error.empty()) {
            std::cout << ",\n      \"reason\": " << quote_json(adapter.device_error);
        }
        std::cout << "\n    }";
        if (index + 1 < adapters.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    std::cout << "  ],\n"
              << "  \"note\": "
              << quote_json(
                     "Device enumeration is not a neural-inference benchmark and does "
                     "not establish physical Arc B580 compatibility.")
              << "\n}\n";
}

std::size_t choose_hardware_adapter(
    const std::vector<AdapterInfo>& adapters,
    const std::string& requested_substring) {
    std::vector<std::size_t> matches;
    const std::string requested = lowercase(requested_substring);

    for (std::size_t index = 0; index < adapters.size(); ++index) {
        const auto& adapter = adapters[index];
        if (adapter.software || !adapter.supports_d3d12) {
            continue;
        }

        const std::string name = lowercase(adapter.name);
        const bool matches_request = !requested.empty()
            ? name.find(requested) != std::string::npos
            : name.find("arc") != std::string::npos;

        if (matches_request) {
            matches.push_back(index);
        }
    }

    if (matches.size() == 1) {
        return matches.front();
    }

    if (requested.empty() && matches.empty()) {
        throw std::runtime_error(
            "No unique Direct3D 12-capable Intel Arc adapter was found. "
            "Use --list-devices, then pass --device-substring with a unique "
            "hardware adapter name. Software adapters are not used for the "
            "hardware self-test.");
    }
    if (matches.empty()) {
        throw std::runtime_error(
            "No Direct3D 12-capable hardware adapter matched --device-substring "
            + quote_json(requested_substring) + ".");
    }
    throw std::runtime_error(
        "The adapter selection matched more than one hardware device. "
        "Choose a longer --device-substring.");
}

std::string shader_model_name(ID3D12Device* device) {
    const D3D_SHADER_MODEL candidates[] = {
        D3D_SHADER_MODEL_6_6,
        D3D_SHADER_MODEL_6_5,
        D3D_SHADER_MODEL_6_4,
        D3D_SHADER_MODEL_6_3,
        D3D_SHADER_MODEL_6_2,
        D3D_SHADER_MODEL_6_1,
        D3D_SHADER_MODEL_6_0,
        D3D_SHADER_MODEL_5_1,
    };

    for (const D3D_SHADER_MODEL candidate : candidates) {
        D3D12_FEATURE_DATA_SHADER_MODEL query{};
        query.HighestShaderModel = candidate;
        if (SUCCEEDED(device->CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL, &query, sizeof(query)))) {
            switch (query.HighestShaderModel) {
            case D3D_SHADER_MODEL_6_6: return "6.6";
            case D3D_SHADER_MODEL_6_5: return "6.5";
            case D3D_SHADER_MODEL_6_4: return "6.4";
            case D3D_SHADER_MODEL_6_3: return "6.3";
            case D3D_SHADER_MODEL_6_2: return "6.2";
            case D3D_SHADER_MODEL_6_1: return "6.1";
            case D3D_SHADER_MODEL_6_0: return "6.0";
            case D3D_SHADER_MODEL_5_1: return "5.1";
            default: break;
            }
        }
    }
    return "unknown";
}

std::string get_blob_text(ID3DBlob* blob) {
    if (blob == nullptr || blob->GetBufferPointer() == nullptr) {
        return {};
    }
    return std::string(
        static_cast<const char*>(blob->GetBufferPointer()),
        blob->GetBufferSize());
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC buffer_description(UINT64 byte_size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Alignment = 0;
    description.Width = byte_size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.SampleDesc.Quality = 0;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    return description;
}

void execute_compute_roundtrip(ID3D12Device* device) {
    const UINT64 byte_size = static_cast<UINT64>(kElementCount) * sizeof(std::uint32_t);

    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_description.NodeMask = 0;

    ComPtr<ID3D12CommandQueue> queue;
    check_hresult(
        device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue)),
        "ID3D12Device::CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> allocator;
    check_hresult(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator)),
        "ID3D12Device::CreateCommandAllocator");

    D3D12_ROOT_PARAMETER root_parameters[2]{};
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    root_parameters[0].Descriptor.ShaderRegister = 0;
    root_parameters[0].Descriptor.RegisterSpace = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[1].Constants.ShaderRegister = 0;
    root_parameters[1].Constants.RegisterSpace = 0;
    root_parameters[1].Constants.Num32BitValues = 1;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_description{};
    root_description.NumParameters = 2;
    root_description.pParameters = root_parameters;
    root_description.NumStaticSamplers = 0;
    root_description.pStaticSamplers = nullptr;
    root_description.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized_root;
    ComPtr<ID3DBlob> root_errors;
    const HRESULT serialize_result = D3D12SerializeRootSignature(
        &root_description,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized_root,
        &root_errors);
    if (FAILED(serialize_result)) {
        throw std::runtime_error(
            "Could not serialize the compute root signature: "
            + get_blob_text(root_errors.Get()));
    }

    ComPtr<ID3D12RootSignature> root_signature;
    check_hresult(
        device->CreateRootSignature(
            0,
            serialized_root->GetBufferPointer(),
            serialized_root->GetBufferSize(),
            IID_PPV_ARGS(&root_signature)),
        "ID3D12Device::CreateRootSignature");

    // This deterministic kernel validates D3D12 dispatch and readback. It is
    // intentionally not presented as part of the TrSS neural network.
    const char* shader_source = R"(
        RWByteAddressBuffer OutputData : register(u0);
        cbuffer Parameters : register(b0) {
            uint ElementCount;
        };

        [numthreads(64, 1, 1)]
        void main(uint3 dispatchId : SV_DispatchThreadID) {
            uint index = dispatchId.x;
            if (index >= ElementCount) {
                return;
            }
            OutputData.Store(index * 4, index * 3 + 7);
        }
    )";

    ComPtr<ID3DBlob> compute_shader;
    ComPtr<ID3DBlob> shader_errors;
    const HRESULT compile_result = D3DCompile(
        shader_source,
        std::char_traits<char>::length(shader_source),
        "trss_dx12_compute_self_test.hlsl",
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &compute_shader,
        &shader_errors);
    if (FAILED(compile_result)) {
        throw std::runtime_error(
            "Could not compile the Direct3D 12 self-test shader: "
            + get_blob_text(shader_errors.Get()));
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_description{};
    pipeline_description.pRootSignature = root_signature.Get();
    pipeline_description.CS.pShaderBytecode = compute_shader->GetBufferPointer();
    pipeline_description.CS.BytecodeLength = compute_shader->GetBufferSize();
    pipeline_description.NodeMask = 0;
    pipeline_description.CachedPSO = {nullptr, 0};
    pipeline_description.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ComPtr<ID3D12PipelineState> pipeline;
    check_hresult(
        device->CreateComputePipelineState(
            &pipeline_description, IID_PPV_ARGS(&pipeline)),
        "ID3D12Device::CreateComputePipelineState");

    const D3D12_HEAP_PROPERTIES default_heap =
        heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_RESOURCE_DESC output_description = buffer_description(
        byte_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ComPtr<ID3D12Resource> output_buffer;
    check_hresult(
        device->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_NONE,
            &output_description,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&output_buffer)),
        "CreateCommittedResource (GPU UAV buffer)");

    const D3D12_HEAP_PROPERTIES readback_heap =
        heap_properties(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC readback_description =
        buffer_description(byte_size, D3D12_RESOURCE_FLAG_NONE);

    ComPtr<ID3D12Resource> readback_buffer;
    check_hresult(
        device->CreateCommittedResource(
            &readback_heap,
            D3D12_HEAP_FLAG_NONE,
            &readback_description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readback_buffer)),
        "CreateCommittedResource (CPU readback buffer)");

    ComPtr<ID3D12GraphicsCommandList> command_list;
    check_hresult(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            allocator.Get(),
            pipeline.Get(),
            IID_PPV_ARGS(&command_list)),
        "ID3D12Device::CreateCommandList");

    command_list->SetComputeRootSignature(root_signature.Get());
    command_list->SetPipelineState(pipeline.Get());
    command_list->SetComputeRootUnorderedAccessView(
        0, output_buffer->GetGPUVirtualAddress());
    command_list->SetComputeRoot32BitConstant(1, kElementCount, 0);
    command_list->Dispatch(
        (kElementCount + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.UAV.pResource = output_buffer.Get();
    command_list->ResourceBarrier(1, &barrier);

    // CopyResource requires its source to be in COPY_SOURCE state. A UAV barrier
    // alone orders UAV accesses but does not perform a resource-state transition.
    D3D12_RESOURCE_BARRIER copy_transition{};
    copy_transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_transition.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    copy_transition.Transition.pResource = output_buffer.Get();
    copy_transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    copy_transition.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    copy_transition.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    command_list->ResourceBarrier(1, &copy_transition);
    command_list->CopyResource(readback_buffer.Get(), output_buffer.Get());

    check_hresult(command_list->Close(), "ID3D12GraphicsCommandList::Close");
    ID3D12CommandList* submitted_lists[] = {command_list.Get()};
    queue->ExecuteCommandLists(1, submitted_lists);

    ComPtr<ID3D12Fence> fence;
    check_hresult(
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
        "ID3D12Device::CreateFence");
    constexpr UINT64 fence_value = 1;
    check_hresult(queue->Signal(fence.Get(), fence_value), "ID3D12CommandQueue::Signal");

    ScopedHandle completion_event(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    if (completion_event.get() == nullptr) {
        throw std::runtime_error("Could not create a fence-completion event.");
    }

    // UINT64_MAX is D3D12's sentinel for a removed device. Never map or trust the
    // readback buffer if the device has been removed or the fence never completes.
    UINT64 completed_value = fence->GetCompletedValue();
    if (completed_value == UINT64_MAX) {
        throw std::runtime_error(
            "The D3D12 device was removed before the compute fence completed ("
            + hresult_text(device->GetDeviceRemovedReason()) + ").");
    }
    if (completed_value < fence_value) {
        check_hresult(
            fence->SetEventOnCompletion(fence_value, completion_event.get()),
            "ID3D12Fence::SetEventOnCompletion");
        const DWORD wait_result = WaitForSingleObject(
            completion_event.get(), kFenceWaitTimeoutMs);
        if (wait_result == WAIT_TIMEOUT) {
            throw std::runtime_error(
                "Timed out after 30 seconds waiting for the D3D12 compute fence. "
                "Device removal reason: "
                + hresult_text(device->GetDeviceRemovedReason()) + ".");
        }
        if (wait_result != WAIT_OBJECT_0) {
            throw std::runtime_error(
                "Waiting for GPU completion failed (Win32 result "
                + std::to_string(wait_result) + ").");
        }
    }

    completed_value = fence->GetCompletedValue();
    if (completed_value == UINT64_MAX) {
        throw std::runtime_error(
            "The D3D12 device was removed while completing the compute self-test ("
            + hresult_text(device->GetDeviceRemovedReason()) + ").");
    }
    if (completed_value < fence_value) {
        throw std::runtime_error(
            "The D3D12 fence event fired before the requested completion value was reached.");
    }

    void* mapped_data = nullptr;
    const D3D12_RANGE read_range{0, static_cast<SIZE_T>(byte_size)};
    check_hresult(
        readback_buffer->Map(
            0, &read_range, reinterpret_cast<void**>(&mapped_data)),
        "ID3D12Resource::Map (readback)");

    const auto* values = static_cast<const std::uint32_t*>(mapped_data);
    for (UINT index = 0; index < kElementCount; ++index) {
        const std::uint32_t expected = index * 3U + 7U;
        if (values[index] != expected) {
            const D3D12_RANGE no_writes{0, 0};
            readback_buffer->Unmap(0, &no_writes);
            throw std::runtime_error(
                "GPU compute self-test returned incorrect data at element "
                + std::to_string(index) + ": expected "
                + std::to_string(expected) + ", got "
                + std::to_string(values[index]) + ".");
        }
    }

    const D3D12_RANGE no_writes{0, 0};
    readback_buffer->Unmap(0, &no_writes);
}


std::string optional_bool(bool available, bool value) {
    if (!available) return "null";
    return value ? "true" : "false";
}

std::string optional_uint(bool available, UINT value) {
    return available ? std::to_string(value) : "null";
}

std::string root_signature_version_name(bool available, D3D_ROOT_SIGNATURE_VERSION version) {
    if (!available) return "unavailable";
    if (version == D3D_ROOT_SIGNATURE_VERSION_1_1) return "1.1";
    if (version == D3D_ROOT_SIGNATURE_VERSION_1_0) return "1.0";
    return "unknown";
}

void print_capability_report(const AdapterInfo& adapter, ID3D12Device* device) {
    // These are documented D3D12 capability queries only. No game process,
    // swap chain, overlay, injection, or anti-cheat interaction is involved.
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    const bool options_available = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    const bool options1_available = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)));

    D3D12_FEATURE_DATA_ROOT_SIGNATURE root_signature{};
    root_signature.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    const bool root_signature_available = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_ROOT_SIGNATURE, &root_signature, sizeof(root_signature)));

    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
    architecture.NodeIndex = 0;
    const bool architecture_available = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture)));

    std::cout << "{\n"
              << "  \"engine\": \"TrSS\",\n"
              << "  \"api\": \"Direct3D 12\",\n"
              << "  \"operation\": \"d3d12_capability_check\",\n"
              << "  \"status\": \"passed\",\n"
              << "  \"adapter\": " << quote_json(adapter.name) << ",\n"
              << "  \"vendor_id\": " << quote_json(hex_id(adapter.vendor_id)) << ",\n"
              << "  \"device_id\": " << quote_json(hex_id(adapter.device_id)) << ",\n"
              << "  \"device_creation_at_feature_level_12_0\": true,\n"
              << "  \"highest_reported_shader_model\": "
              << quote_json(shader_model_name(device)) << ",\n"
              << "  \"root_signature_query_available\": "
              << (root_signature_available ? "true" : "false") << ",\n"
              << "  \"root_signature_version\": "
              << quote_json(root_signature_version_name(
                     root_signature_available, root_signature.HighestVersion)) << ",\n"
              << "  \"options_query_available\": "
              << (options_available ? "true" : "false") << ",\n"
              << "  \"resource_binding_tier\": "
              << optional_uint(options_available, static_cast<UINT>(options.ResourceBindingTier)) << ",\n"
              << "  \"resource_heap_tier\": "
              << optional_uint(options_available, static_cast<UINT>(options.ResourceHeapTier)) << ",\n"
              << "  \"tiled_resources_tier\": "
              << optional_uint(options_available, static_cast<UINT>(options.TiledResourcesTier)) << ",\n"
              << "  \"typed_uav_load_additional_formats\": "
              << optional_bool(options_available, options.TypedUAVLoadAdditionalFormats) << ",\n"
              << "  \"options1_query_available\": "
              << (options1_available ? "true" : "false") << ",\n"
              << "  \"wave_operations_supported\": "
              << optional_bool(options1_available, options1.WaveOps) << ",\n"
              << "  \"wave_lane_count_min\": "
              << optional_uint(options1_available, options1.WaveLaneCountMin) << ",\n"
              << "  \"wave_lane_count_max\": "
              << optional_uint(options1_available, options1.WaveLaneCountMax) << ",\n"
              << "  \"total_wave_lane_count\": "
              << optional_uint(options1_available, options1.TotalLaneCount) << ",\n"
              << "  \"int64_shader_ops_supported\": "
              << optional_bool(options1_available, options1.Int64ShaderOps) << ",\n"
              << "  \"architecture_query_available\": "
              << (architecture_available ? "true" : "false") << ",\n"
              << "  \"uma\": "
              << optional_bool(architecture_available, architecture.UMA) << ",\n"
              << "  \"cache_coherent_uma\": "
              << optional_bool(architecture_available, architecture.CacheCoherentUMA) << ",\n"
              << "  \"tile_based_renderer\": "
              << optional_bool(architecture_available, architecture.TileBasedRenderer) << ",\n"
              << "  \"note\": "
              << quote_json(
                     "This is a standalone API capability report, not a game-compatibility "
                     "test or benchmark. It does not attach to or modify any game, inject a "
                     "DLL, or interact with anti-cheat. A passing D3D12 check does not mean "
                     "the separate TrSS D3D11 live runtime is compatible.")
              << "\n}\n";
}

void print_self_test_report(
    const AdapterInfo& adapter,
    ID3D12Device* device,
    bool warp_test) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    const bool options1_available = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)));
    const bool wave_ops = options1_available && options1.WaveOps;

    execute_compute_roundtrip(device);

    std::cout << "{\n"
              << "  \"engine\": \"TrSS\",\n"
              << "  \"backend\": "
              << quote_json(warp_test
                     ? "Direct3D 12 compute (WARP software adapter)"
                     : "Direct3D 12 compute (hardware adapter)") << ",\n"
              << "  \"status\": \"passed\",\n"
              << "  \"adapter\": " << quote_json(adapter.name) << ",\n"
              << "  \"vendor_id\": " << quote_json(hex_id(adapter.vendor_id)) << ",\n"
              << "  \"device_id\": " << quote_json(hex_id(adapter.device_id)) << ",\n"
              << "  \"minimum_feature_level\": \"12_0\",\n"
              << "  \"highest_reported_shader_model\": "
              << quote_json(shader_model_name(device)) << ",\n"
              << "  \"wave_operations_supported\": "
              << (wave_ops ? "true" : "false") << ",\n"
              << "  \"test\": \"hlsl_compute_uav_fence_readback\",\n"
              << "  \"verified_elements\": " << kElementCount << ",\n"
              << "  \"result_rule\": \"value[i] = i * 3 + 7\",\n"
              << "  \"note\": "
              << quote_json(
                     "This validates the Direct3D 12 compute path only. It does not "
                     "run TrSS neural inference, load trained weights, or establish "
                     "Arc B580 performance.")
              << "\n}\n";
}

AdapterInfo create_warp_adapter() {
    const auto factory = create_factory();
    ComPtr<IDXGIAdapter1> warp;
    check_hresult(
        factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)),
        "IDXGIFactory4::EnumWarpAdapter");

    AdapterInfo info = describe_adapter(std::move(warp));
    if (!info.supports_d3d12) {
        throw std::runtime_error(
            "The WARP software adapter does not expose Direct3D 12 feature level 12_0.");
    }
    return info;
}

void print_usage() {
    std::cout
        << "TrSS Direct3D 12 compute bring-up\n\n"
        << "Usage:\n"
        << "  trss_dx12_probe.exe --list-devices\n"
        << "  trss_dx12_probe.exe --capability-check [--device-substring NAME]\n"
        << "  trss_dx12_probe.exe --self-test [--device-substring NAME]\n"
        << "  trss_dx12_probe.exe --warp-self-test\n\n"
        << "The capability check queries the selected hardware adapter without opening a game.\n"
        << "The hardware self-test runs a compute/readback roundtrip on one uniquely selected "
           "D3D12-capable adapter whose name contains 'Arc', unless --device-substring is given.\n"
        << "The WARP test uses a software adapter for CI and is not GPU performance "
           "validation. Choose exactly one operation.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool list_devices = false;
        bool capability_check = false;
        bool self_test = false;
        bool warp_self_test = false;
        std::string device_substring;

        for (int index = 1; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--help" || argument == "-h") {
                print_usage();
                return 0;
            } else if (argument == "--list-devices") {
                list_devices = true;
            } else if (argument == "--capability-check") {
                capability_check = true;
            } else if (argument == "--self-test") {
                self_test = true;
            } else if (argument == "--warp-self-test") {
                warp_self_test = true;
            } else if (argument == "--device-substring") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--device-substring needs a device-name value.");
                }
                device_substring = argv[++index];
                if (device_substring.empty()) {
                    throw std::runtime_error("--device-substring cannot be empty.");
                }
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        const int selected_modes =
            static_cast<int>(list_devices) + static_cast<int>(capability_check)
            + static_cast<int>(self_test) + static_cast<int>(warp_self_test);
        if (selected_modes != 1) {
            throw std::runtime_error(
                "Choose exactly one of --list-devices, --capability-check, --self-test, "
                "or --warp-self-test.");
        }
        if (!device_substring.empty() && (list_devices || warp_self_test)) {
            throw std::runtime_error(
                "--device-substring is only valid with --capability-check or the hardware --self-test.");
        }

        if (list_devices) {
            print_adapter_list(enumerate_adapters());
            return 0;
        }

        if (warp_self_test) {
            AdapterInfo adapter = create_warp_adapter();
            ComPtr<ID3D12Device> device;
            check_hresult(
                D3D12CreateDevice(
                    adapter.adapter.Get(),
                    D3D_FEATURE_LEVEL_12_0,
                    IID_PPV_ARGS(&device)),
                "D3D12CreateDevice (WARP)");
            print_self_test_report(adapter, device.Get(), true);
            return 0;
        }

        const auto adapters = enumerate_adapters();
        const std::size_t selected_index =
            choose_hardware_adapter(adapters, device_substring);
        const AdapterInfo& adapter = adapters[selected_index];

        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.adapter.Get(),
                D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice (hardware)");

        if (capability_check) {
            print_capability_report(adapter, device.Get());
        } else {
            print_self_test_report(adapter, device.Get(), false);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TrSS Direct3D 12 error: " << error.what() << '\n';
        return 2;
    }
}
