// First-party color-only desktop sampler. It does not inject into games or expose depth/motion.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr wchar_t kMarker[] = L".trss-desktop-capture-session";
constexpr std::uint64_t kMaxFrameBytes = 512ULL * 1024ULL * 1024ULL;

struct Options {
    bool list = false, force = false, help = false, has_out = false, has_test = false;
    fs::path out, test_png;
    int adapter = 0, monitor = 0, fps = 30, max_frames = 300;
    double seconds = 5.0;
};
struct OutputChoice {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<IDXGIOutput> output;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    DXGI_OUTPUT_DESC output_desc{};
};

static void hr_check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::ostringstream out;
        out << what << " failed with HRESULT 0x" << std::hex << std::uppercase
            << static_cast<unsigned long>(hr);
        throw std::runtime_error(out.str());
    }
}
static std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (count <= 0) return {};
    std::string result(static_cast<size_t>(count), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), count, nullptr, nullptr);
    return result;
}
static std::string escaped(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) out << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0') << static_cast<unsigned int>(ch);
            else out << static_cast<char>(ch);
        }
    }
    return out.str();
}
static std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_s(&tm, &t);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setw(3) << std::setfill('0') << ms.count() << 'Z';
    return out.str();
}
static std::wstring value_for(int& i, int argc, wchar_t** argv, const wchar_t* flag) {
    if (i + 1 >= argc) {
        std::wcerr << L"Missing value for " << flag << L".\n";
        throw std::runtime_error("A command-line value is missing.");
    }
    return argv[++i];
}
static int parse_int(const std::wstring& text, const wchar_t* name) {
    size_t used = 0;
    int value = 0;
    try { value = std::stoi(text, &used); }
    catch (...) { throw std::runtime_error("Invalid integer command-line argument."); }
    if (used != text.size()) {
        std::wcerr << L"Invalid integer for " << name << L".\n";
        throw std::runtime_error("Invalid integer command-line argument.");
    }
    return value;
}
static double parse_double(const std::wstring& text) {
    size_t used = 0;
    double value = 0.0;
    try { value = std::stod(text, &used); }
    catch (...) { throw std::runtime_error("Invalid seconds argument."); }
    if (used != text.size() || !std::isfinite(value))
        throw std::runtime_error("--seconds must be finite.");
    return value;
}
static Options parse(int argc, wchar_t** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::wstring arg = argv[i];
        if (arg == L"--help" || arg == L"-h") o.help = true;
        else if (arg == L"--list-devices") o.list = true;
        else if (arg == L"--force") o.force = true;
        else if (arg == L"--output-dir") { o.out = value_for(i,argc,argv,L"--output-dir"); o.has_out=true; }
        else if (arg == L"--self-test-png") { o.test_png=value_for(i,argc,argv,L"--self-test-png"); o.has_test=true; }
        else if (arg == L"--adapter") o.adapter=parse_int(value_for(i,argc,argv,L"--adapter"),L"--adapter");
        else if (arg == L"--monitor") o.monitor=parse_int(value_for(i,argc,argv,L"--monitor"),L"--monitor");
        else if (arg == L"--fps") o.fps=parse_int(value_for(i,argc,argv,L"--fps"),L"--fps");
        else if (arg == L"--max-frames") o.max_frames=parse_int(value_for(i,argc,argv,L"--max-frames"),L"--max-frames");
        else if (arg == L"--seconds") o.seconds=parse_double(value_for(i,argc,argv,L"--seconds"));
        else { std::wcerr << L"Unknown option: " << arg << L"\n"; throw std::runtime_error("Unknown option."); }
    }
    if (o.adapter < 0 || o.monitor < 0) throw std::runtime_error("--adapter and --monitor must be non-negative.");
    if (o.fps < 1 || o.fps > 240) throw std::runtime_error("--fps must be 1 through 240.");
    if (o.max_frames < 1 || o.max_frames > 36000) throw std::runtime_error("--max-frames must be 1 through 36000.");
    if (o.seconds < 0.1 || o.seconds > 600.0) throw std::runtime_error("--seconds must be 0.1 through 600.");
    if (o.has_test && (o.has_out || o.force)) throw std::runtime_error("--self-test-png cannot be combined with capture output options.");
    if (!o.help && !o.list && !o.has_test && !o.has_out)
        throw std::runtime_error("Specify --output-dir, --list-devices, --self-test-png, or --help.");
    if (o.list && (o.has_out || o.has_test || o.force))
        throw std::runtime_error("--list-devices cannot be combined with output options.");
    return o;
}
static void help() {
    std::wcout
      << L"TrSS Desktop Capture - DXGI Desktop Duplication sampler\n\n"
      << L"  --list-devices              List adapter and monitor indexes\n"
      << L"  --output-dir <folder>       New capture folder (required to record)\n"
      << L"  --adapter <index>           DXGI adapter index (default 0)\n"
      << L"  --monitor <index>           Output index on that adapter (default 0)\n"
      << L"  --seconds <n>               0.1 to 600 seconds (default 5)\n"
      << L"  --fps <n>                   1 to 240 saved samples/sec (default 30)\n"
      << L"  --max-frames <n>            1 to 36000 frame cap (default 300)\n"
      << L"  --force                     Replace a previous TrSS capture folder only\n"
      << L"  --self-test-png <file>      Test PNG encoding without desktop capture\n\n"
      << L"This is color-only desktop capture. It does not read game depth, motion,\n"
      << L"history, ground-truth renders, or private render targets.\n";
}
static ComPtr<IDXGIFactory1> factory() {
    ComPtr<IDXGIFactory1> f;
    hr_check(CreateDXGIFactory1(IID_PPV_ARGS(&f)), "CreateDXGIFactory1");
    return f;
}
static void list_devices() {
    const auto f = factory();
    bool any = false;
    for (UINT a = 0;; ++a) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT hr = f->EnumAdapters1(a, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        hr_check(hr, "Enumerate adapter");
        DXGI_ADAPTER_DESC1 ad{};
        hr_check(adapter->GetDesc1(&ad), "Get adapter description");
        any = true;
        std::wcout << L"Adapter " << a << L": " << ad.Description
                   << L" (vendor 0x" << std::hex << ad.VendorId
                   << L", device 0x" << ad.DeviceId << std::dec << L")\n";
        for (UINT m = 0;; ++m) {
            ComPtr<IDXGIOutput> output;
            hr = adapter->EnumOutputs(m, &output);
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            hr_check(hr, "Enumerate monitor");
            DXGI_OUTPUT_DESC od{};
            hr_check(output->GetDesc(&od), "Get monitor description");
            std::wcout << L"  Monitor " << m << L": " << od.DeviceName << L" "
                << (od.DesktopCoordinates.right-od.DesktopCoordinates.left) << L"x"
                << (od.DesktopCoordinates.bottom-od.DesktopCoordinates.top)
                << L", attached=" << (od.AttachedToDesktop ? L"yes" : L"no")
                << L", origin=(" << od.DesktopCoordinates.left << L","
                << od.DesktopCoordinates.top << L")\n";
        }
    }
    if (!any) std::wcout << L"No DXGI adapters reported by this system.\n";
}
static OutputChoice choose_output(int adapter_index, int monitor_index) {
    const auto f = factory();
    OutputChoice selected;
    hr_check(f->EnumAdapters1(static_cast<UINT>(adapter_index), &selected.adapter), "Select adapter");
    hr_check(selected.adapter->GetDesc1(&selected.adapter_desc), "Get selected adapter description");
    hr_check(selected.adapter->EnumOutputs(static_cast<UINT>(monitor_index), &selected.output), "Select monitor");
    hr_check(selected.output->GetDesc(&selected.output_desc), "Get selected monitor description");
    if (!selected.output_desc.AttachedToDesktop)
        throw std::runtime_error("Selected output is not attached to the desktop; use --list-devices.");
    return selected;
}
static ComPtr<IWICImagingFactory> wic_factory() {
    ComPtr<IWICImagingFactory> f;
    hr_check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&f)), "Create WIC encoder factory");
    return f;
}
static void write_png(IWICImagingFactory* factory, const fs::path& path,
                      UINT width, UINT height, const std::vector<std::uint8_t>& pixels) {
    const std::uint64_t stride = static_cast<std::uint64_t>(width) * 4U;
    const std::uint64_t bytes = stride * height;
    if (!width || !height || stride > UINT_MAX || bytes > UINT_MAX || bytes != pixels.size())
        throw std::runtime_error("Invalid PNG dimensions or BGRA buffer size.");
    ComPtr<IWICStream> stream;
    hr_check(factory->CreateStream(&stream), "Create WIC stream");
    hr_check(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), "Open PNG output");
    ComPtr<IWICBitmapEncoder> encoder;
    hr_check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "Create PNG encoder");
    hr_check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "Initialize PNG encoder");
    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> props;
    hr_check(encoder->CreateNewFrame(&frame, &props), "Create PNG frame");
    hr_check(frame->Initialize(props.Get()), "Initialize PNG frame");
    hr_check(frame->SetSize(width, height), "Set PNG dimensions");
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    hr_check(frame->SetPixelFormat(&fmt), "Set PNG pixel format");
    if (fmt != GUID_WICPixelFormat32bppBGRA)
        throw std::runtime_error("WIC encoder did not accept 32-bit BGRA pixels.");
    hr_check(frame->WritePixels(height, static_cast<UINT>(stride), static_cast<UINT>(bytes),
        const_cast<BYTE*>(reinterpret_cast<const BYTE*>(pixels.data()))), "Write PNG pixels");
    hr_check(frame->Commit(), "Commit PNG frame");
    hr_check(encoder->Commit(), "Commit PNG file");
}
static void self_test_png(const fs::path& requested) {
    const fs::path path = fs::absolute(requested).lexically_normal();
    if (fs::exists(path)) throw std::runtime_error("PNG self-test will not overwrite an existing file.");
    fs::create_directories(path.parent_path());
    constexpr UINT w=64, h=36;
    std::vector<std::uint8_t> pixels(w*h*4U, 255U);
    for (UINT y=0; y<h; ++y) for (UINT x=0; x<w; ++x) {
        const size_t i=(static_cast<size_t>(y)*w+x)*4U;
        pixels[i]=static_cast<std::uint8_t>((x*5U)&255U);
        pixels[i+1]=static_cast<std::uint8_t>((y*7U)&255U);
        pixels[i+2]=static_cast<std::uint8_t>(((x+y)*3U)&255U);
        pixels[i+3]=255U;
    }
    auto f=wic_factory();
    write_png(f.Get(), path, w, h, pixels);
    std::ifstream input(path, std::ios::binary);
    const std::uint8_t expected[8]={137,80,78,71,13,10,26,10};
    std::uint8_t actual[8]{};
    input.read(reinterpret_cast<char*>(actual),8);
    if (!input) throw std::runtime_error("Could not verify the PNG self-test output.");
    for (size_t i=0;i<8;++i) if (actual[i]!=expected[i])
        throw std::runtime_error("PNG signature verification failed.");
    std::wcout << L"PNG encoder self-test passed: " << path.wstring() << L"\n";
}

class OutputTransaction {
public:
    OutputTransaction(const fs::path& requested, bool force) {
        output_=fs::absolute(requested).lexically_normal();
        const auto cwd=fs::current_path();
        if (output_.filename().empty() || output_==output_.root_path() || output_==cwd)
            throw std::runtime_error("Output must be a dedicated folder, not a root or current folder.");
        const auto cwd_from_output=cwd.lexically_relative(output_);
        if (!cwd_from_output.empty() && *cwd_from_output.begin()!=fs::path(L".."))
            throw std::runtime_error("Output cannot be a parent of the current working folder.");
        if (fs::exists(output_) && !fs::is_directory(output_))
            throw std::runtime_error("Output path exists as a file.");
        if (fs::exists(output_) && !force)
            throw std::runtime_error("Output folder exists. Use --force only for a TrSS capture folder.");
        if (fs::exists(output_) && !fs::is_regular_file(output_/kMarker))
            throw std::runtime_error("--force is allowed only for a marked TrSS capture folder.");
        fs::create_directories(output_.parent_path());
        const auto nonce=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        staging_=output_.parent_path()/(output_.filename().wstring()+L".staging-"+std::to_wstring(nonce));
        fs::create_directories(staging_/L"frames");
    }
    ~OutputTransaction() {
        std::error_code ignored;
        if (!committed_) fs::remove_all(staging_,ignored);
        if (!backup_.empty() && fs::exists(backup_) && !fs::exists(output_))
            fs::rename(backup_,output_,ignored);
    }
    const fs::path& staging() const { return staging_; }
    void commit(bool force) {
        bool moved=false;
        try {
            if (fs::exists(output_)) {
                if (!force) throw std::runtime_error("Output folder appeared during capture.");
                const auto nonce=std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                backup_=output_.parent_path()/(output_.filename().wstring()+L".backup-"+std::to_wstring(nonce));
                fs::rename(output_,backup_);
                moved=true;
            }
            fs::rename(staging_,output_);
            committed_=true;
        } catch (...) {
            if (moved && fs::exists(backup_) && !fs::exists(output_)) {
                std::error_code ignored;
                fs::rename(backup_,output_,ignored);
                backup_.clear();
            }
            throw;
        }
        if (!backup_.empty()) { std::error_code ignored; fs::remove_all(backup_,ignored); backup_.clear(); }
    }
private:
    fs::path output_,staging_,backup_;
    bool committed_=false;
};

static bool acquire_latest(IDXGIOutputDuplication* duplication, ID3D11Device* device,
    ID3D11DeviceContext* context, int wait_ms, UINT w, UINT h,
    ComPtr<ID3D11Texture2D>& staging, std::vector<std::uint8_t>& pixels) {
    DXGI_OUTDUPL_FRAME_INFO info{};
    ComPtr<IDXGIResource> resource;
    const HRESULT acquired=duplication->AcquireNextFrame(static_cast<UINT>(wait_ms),&info,&resource);
    if (acquired==DXGI_ERROR_WAIT_TIMEOUT) return false;
    if (acquired==DXGI_ERROR_ACCESS_LOST)
        throw std::runtime_error("Desktop access lost after a display change; restart capture.");
    hr_check(acquired,"Acquire desktop frame");
    bool released=false;
    try {
        ComPtr<ID3D11Texture2D> source;
        hr_check(resource.As(&source),"Query desktop texture");
        D3D11_TEXTURE2D_DESC d{};
        source->GetDesc(&d);
        if (d.Format!=DXGI_FORMAT_B8G8R8A8_UNORM)
            throw std::runtime_error("Unexpected desktop pixel format; BGRA8 was expected.");
        if (d.Width!=w || d.Height!=h)
            throw std::runtime_error("Desktop resolution changed during capture; restart the session.");
        const std::uint64_t bytes=static_cast<std::uint64_t>(w)*h*4U;
        if (bytes>kMaxFrameBytes) throw std::runtime_error("Frame readback exceeds the 512 MiB safety limit.");
        if (!staging) {
            d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0;
            d.CPUAccessFlags=D3D11_CPU_ACCESS_READ; d.MiscFlags=0;
            hr_check(device->CreateTexture2D(&d,nullptr,&staging),"Create readback texture");
        }
        context->CopyResource(staging.Get(),source.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr_check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"Map readback texture");
        try {
            pixels.resize(static_cast<size_t>(bytes));
            const size_t row=static_cast<size_t>(w)*4U;
            const auto* source_bytes=static_cast<const std::uint8_t*>(mapped.pData);
            for (UINT y=0;y<h;++y)
                std::memcpy(pixels.data()+static_cast<size_t>(y)*row,
                            source_bytes+static_cast<size_t>(y)*mapped.RowPitch,row);
        } catch (...) { context->Unmap(staging.Get(),0); throw; }
        context->Unmap(staging.Get(),0);
        const HRESULT release=duplication->ReleaseFrame();
        released=true;
        hr_check(release,"Release desktop frame");
        return true;
    } catch (...) {
        if (!released) duplication->ReleaseFrame();
        throw;
    }
}
static void write_report(const fs::path& root, const OutputChoice& choice, const Options& o,
    const std::string& start_utc, const std::string& end_utc, UINT w, UINT h,
    int frames, int skipped, double elapsed) {
    std::ofstream report(root/L"session.json",std::ios::binary|std::ios::out);
    if (!report) throw std::runtime_error("Could not create session.json.");
    report << "{\n"
      << "  \"format\":\"trss-desktop-capture-session-v1\",\n"
      << "  \"capture_api\":\"DXGI Desktop Duplication with Direct3D 11 CPU readback\",\n"
      << "  \"started_utc\":\"" << escaped(start_utc) << "\",\n"
      << "  \"finished_utc\":\"" << escaped(end_utc) << "\",\n"
      << "  \"adapter_index\":" << o.adapter << ",\n"
      << "  \"adapter_description\":\"" << escaped(utf8(choice.adapter_desc.Description)) << "\",\n"
      << "  \"monitor_index\":" << o.monitor << ",\n"
      << "  \"monitor_device_name\":\"" << escaped(utf8(choice.output_desc.DeviceName)) << "\",\n"
      << "  \"desktop_rect\":[" << choice.output_desc.DesktopCoordinates.left << ","
        << choice.output_desc.DesktopCoordinates.top << "," << choice.output_desc.DesktopCoordinates.right
        << "," << choice.output_desc.DesktopCoordinates.bottom << "],\n"
      << "  \"width\":" << w << ",\n  \"height\":" << h << ",\n"
      << "  \"requested_fps\":" << o.fps << ",\n  \"frames_saved\":" << frames << ",\n"
      << "  \"sample_slots_skipped\":" << skipped << ",\n"
      << "  \"elapsed_seconds\":" << std::fixed << std::setprecision(6) << elapsed << ",\n"
      << "  \"measured_output_fps\":" << (elapsed>0.0 ? frames/elapsed : 0.0) << ",\n"
      << "  \"manifest\":\"capture-manifest.jsonl\",\n"
      << "  \"has_depth\":false,\n  \"has_motion_vectors\":false,\n"
      << "  \"has_high_resolution_reference\":false,\n  \"training_ready\":false,\n"
      << "  \"limits\":[\"Color only; this does not read a game's private render targets.\","
      << "\"No depth, motion, disocclusion mask, or paired reference is captured.\","
      << "\"CPU readback sampling is not a real-time hook or game-FPS benchmark.\","
      << "\"Only use this sequence with model adapters that do not need missing auxiliary inputs.\"]\n}\n";
    report.flush();
    if (!report) throw std::runtime_error("Could not finish session.json.");
}
static void capture(const Options& o) {
    const OutputChoice choice=choose_output(o.adapter,o.monitor);
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL selected_level{};
    const D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,D3D_FEATURE_LEVEL_10_0};
    HRESULT result=D3D11CreateDevice(choice.adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,4,D3D11_SDK_VERSION,&device,&selected_level,&context);
    if (result==E_INVALIDARG)
        result=D3D11CreateDevice(choice.adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels+1,3,D3D11_SDK_VERSION,
            &device,&selected_level,&context);
    hr_check(result,"Create D3D11 capture device");
    ComPtr<IDXGIOutput1> output1;
    hr_check(choice.output.As(&output1),"Query IDXGIOutput1");
    ComPtr<IDXGIOutputDuplication> duplication;
    hr_check(output1->DuplicateOutput(device.Get(),&duplication),"Duplicate selected output");
    DXGI_OUTDUPL_DESC desc{};
    duplication->GetDesc(&desc);
    const UINT w=desc.ModeDesc.Width,h=desc.ModeDesc.Height;
    if (!w || !h) throw std::runtime_error("The selected monitor has an empty display mode.");

    OutputTransaction transaction(o.out,o.force);
    const fs::path& root=transaction.staging();
    std::ofstream manifest(root/L"capture-manifest.jsonl",std::ios::binary|std::ios::out);
    if (!manifest) throw std::runtime_error("Could not create capture manifest.");
    auto wic=wic_factory();
    ComPtr<ID3D11Texture2D> readback;
    std::vector<std::uint8_t> latest;
    const auto start=Clock::now();
    const auto deadline=start+std::chrono::duration<double>(o.seconds);
    const auto interval=std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0/o.fps));
    auto next_sample=start;
    const std::string start_utc=utc_now();
    int frames=0,skipped=0;
    bool have_pixels=false;
    std::wcout << L"Sampling " << w << L"x" << h << L" from "
      << choice.adapter_desc.Description << L" / " << choice.output_desc.DeviceName << L"\n";

    while (Clock::now()<deadline && frames<o.max_frames) {
        auto now=Clock::now();
        if (have_pixels && now>=next_sample) {
            const double timestamp=std::chrono::duration<double>(now-start).count();
            std::wostringstream name;
            name << L"frame_" << std::setw(6) << std::setfill(L'0') << frames << L".png";
            write_png(wic.Get(),root/L"frames"/name.str(),w,h,latest);
            const std::string relative="frames/"+utf8(name.str());
            manifest << "{\"frame_id\":" << frames << ",\"timestamp\":"
              << std::fixed << std::setprecision(6) << timestamp
              << ",\"color\":\"" << escaped(relative)
              << "\",\"scene_id\":\"desktop-capture-session\",\"camera_cut\":"
              << (frames==0 ? "true" : "false")
              << ",\"color_space\":\"unknown\",\"depth_encoding\":\"unknown\","
              << "\"motion_units\":\"unknown\"}\n";
            if (!manifest) throw std::runtime_error("Could not write a manifest row.");
            ++frames;
            next_sample+=interval;
            const auto after=Clock::now();
            while (next_sample<=after) { next_sample+=interval; ++skipped; }
            continue;
        }
        now=Clock::now();
        if (now>=deadline || frames>=o.max_frames) break;
        int wait_ms=8;
        if (have_pixels && next_sample>now) {
            auto left=std::chrono::duration_cast<std::chrono::milliseconds>(next_sample-now).count();
            wait_ms=static_cast<int>(std::clamp<long long>(left,1,8));
        }
        have_pixels=acquire_latest(duplication.Get(),device.Get(),context.Get(),wait_ms,w,h,
            readback,latest) || have_pixels;
    }
    manifest.flush();
    if (!manifest) throw std::runtime_error("Could not flush capture manifest.");
    manifest.close();
    if (!frames) throw std::runtime_error("No desktop frame was acquired; check session/monitor.");
    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    {
        std::ofstream marker(root/kMarker,std::ios::binary|std::ios::out);
        marker << "Created by TrSS Desktop Capture. Safe for --force replacement.\n";
        if (!marker) throw std::runtime_error("Could not write capture safety marker.");
    }
    write_report(root,choice,o,start_utc,utc_now(),w,h,frames,skipped,elapsed);
    transaction.commit(o.force);
    std::wcout << L"Saved " << frames << L" color frames to "
      << fs::absolute(o.out).lexically_normal().wstring() << L"\n"
      << L"Depth and motion vectors were not captured.\n";
}

int wmain(int argc, wchar_t** argv) {
    const HRESULT com=CoInitializeEx(nullptr,COINIT_MULTITHREADED);
    if (FAILED(com)) {
        std::wcerr << L"COM initialization failed (HRESULT 0x" << std::hex
                   << static_cast<unsigned long>(com) << L").\n";
        return 2;
    }
    int code=0;
    try {
        const Options o=parse(argc,argv);
        if (o.help) help();
        else if (o.list) list_devices();
        else if (o.has_test) self_test_png(o.test_png);
        else capture(o);
    } catch (const std::exception& e) {
        std::cerr << "TrSS Desktop Capture error: " << e.what() << "\n";
        code=1;
    } catch (...) {
        std::cerr << "TrSS Desktop Capture error: unexpected failure.\n";
        code=1;
    }
    CoUninitialize();
    return code;
}
