// TrSS Sample Recorder: a native Windows GUI that captures visible game-window RGB samples.
// It does not inject into games, install an overlay, read private renderer buffers, or upload files.
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>
#include <knownfolders.h>
#include <gdiplus.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <objidl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <regex>
#include <ctime>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "shell32.lib")

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace Gdiplus;

namespace {
constexpr std::uint64_t kMaximumJsonBytes = 23ULL * 1024ULL * 1024ULL;
constexpr int kWarmupSeconds = 120;
constexpr int kSampleIntervalMs = 500;
constexpr int kMaximumImageDimension = 1920;
constexpr UINT_PTR kTimerId = 1;
constexpr UINT kTimerIntervalMs = 250;

enum ControlId {
    IdGameList = 101,
    IdScan = 102,
    IdLaunch = 103,
    IdChooseFolder = 104,
    IdOpenFolder = 105,
    IdAutoCapture = 106
};

struct GameEntry {
    std::wstring name;
    std::wstring source;
    fs::path installPath;
    std::wstring launchTarget;
    bool launchIsUri = false;
};

struct WindowCandidate {
    HWND window = nullptr;
    DWORD processId = 0;
    int gameIndex = -1;
    std::wstring executablePath;
};

struct FrameTelemetry {
    bool hasCpu = false;
    double cpuPercent = 0.0;
    bool hasWorkingSet = false;
    std::uint64_t workingSetBytes = 0;
};

struct CapturedImage {
    std::vector<unsigned char> jpeg;
    int sourceWidth = 0;
    int sourceHeight = 0;
    int width = 0;
    int height = 0;
};

HINSTANCE g_instance = nullptr;
HWND g_window = nullptr;
HWND g_heading = nullptr;
HWND g_subheading = nullptr;
HWND g_gameGroup = nullptr;
HWND g_gameList = nullptr;
HWND g_detailGroup = nullptr;
HWND g_activeGame = nullptr;
HWND g_status = nullptr;
HWND g_progress = nullptr;
HWND g_details = nullptr;
HWND g_outputText = nullptr;
HWND g_scanButton = nullptr;
HWND g_launchButton = nullptr;
HWND g_chooseButton = nullptr;
HWND g_openButton = nullptr;
HWND g_autoCapture = nullptr;
HFONT g_font = nullptr;
HFONT g_headingFont = nullptr;
HBRUSH g_backgroundBrush = nullptr;
std::vector<GameEntry> g_games;
fs::path g_outputRoot;
std::wstring g_statusCache;
std::wstring g_activeCache;
std::wstring g_detailsCache;
std::wstring g_outputCache;
ULONG_PTR g_gdiplusToken = 0;
DWORD g_trackedPid = 0;
int g_trackedGameIndex = -1;
HWND g_trackedWindow = nullptr;
std::wstring g_trackedExecutable;
Clock::time_point g_trackedAt{};
Clock::time_point g_lastDetection{};
Clock::time_point g_lastCapture{};
std::wstring g_sessionId;
std::uint64_t g_frameIndex = 0;
bool g_trackingDetailsDirty = true;
bool g_captureErrorShown = false;

std::string to_utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int length = WideCharToMultiByte(CP_UTF8, 0, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (length <= 0) return {};
    std::string result(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), length, nullptr, nullptr);
    return result;
}

std::wstring from_utf8(const std::string& value) {
    if (value.empty()) return {};
    int length = MultiByteToWideChar(CP_UTF8, 0, value.data(),
        static_cast<int>(value.size()), nullptr, 0);
    UINT page = CP_UTF8;
    if (length <= 0) {
        page = CP_ACP;
        length = MultiByteToWideChar(page, 0, value.data(),
            static_cast<int>(value.size()), nullptr, 0);
    }
    if (length <= 0) return {};
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(page, 0, value.data(), static_cast<int>(value.size()),
        result.data(), length);
    return result;
}

std::string json_escape(const std::string& value) {
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
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch) << std::dec;
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

std::wstring lower(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); });
    return value;
}

std::wstring trim(std::wstring value) {
    const auto first = value.find_first_not_of(L" \t\r\n");
    if (first == std::wstring::npos) return {};
    const auto last = value.find_last_not_of(L" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::wstring normalized_token(std::wstring value) {
    value = lower(std::move(value));
    value.erase(std::remove_if(value.begin(), value.end(),
        [](wchar_t ch) { return !std::iswalnum(ch); }), value.end());
    return value;
}

void replace_all(std::wstring& value, const std::wstring& from, const std::wstring& to) {
    if (from.empty()) return;
    std::size_t offset = 0;
    while ((offset = value.find(from, offset)) != std::wstring::npos) {
        value.replace(offset, from.size(), to);
        offset += to.size();
    }
}

std::wstring read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    const std::string bytes((std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    return from_utf8(bytes);
}

std::wstring extract_vdf_value(const std::wstring& text, const std::wstring& key) {
    const std::wregex expression(L"\"" + key + L"\"\\s*\"([^\"]*)\"",
        std::regex_constants::icase);
    std::wsmatch match;
    if (!std::regex_search(text, match, expression)) return {};
    std::wstring value = match[1].str();
    replace_all(value, L"\\\\", L"\\");
    return value;
}

std::wstring extract_json_value(const std::wstring& text, const std::wstring& key) {
    const std::wregex expression(L"\"" + key +
        L"\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"",
        std::regex_constants::icase);
    std::wsmatch match;
    if (!std::regex_search(text, match, expression)) return {};
    const std::wstring raw = match[1].str();
    std::wstring value;
    value.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] != L'\\' || i + 1 >= raw.size()) {
            value.push_back(raw[i]);
            continue;
        }
        const wchar_t escaped = raw[++i];
        switch (escaped) {
        case L'"': value.push_back(L'"'); break;
        case L'\\': value.push_back(L'\\'); break;
        case L'/': value.push_back(L'/'); break;
        case L'b': value.push_back(L'\b'); break;
        case L'f': value.push_back(L'\f'); break;
        case L'n': value.push_back(L'\n'); break;
        case L'r': value.push_back(L'\r'); break;
        case L't': value.push_back(L'\t'); break;
        case L'u':
            if (i + 4 < raw.size()) {
                wchar_t codepoint = 0;
                bool valid = true;
                for (int digit = 0; digit < 4; ++digit) {
                    const wchar_t hex = raw[i + 1 + digit];
                    codepoint = static_cast<wchar_t>(codepoint << 4);
                    if (hex >= L'0' && hex <= L'9') codepoint += hex - L'0';
                    else if (hex >= L'a' && hex <= L'f') codepoint += hex - L'a' + 10;
                    else if (hex >= L'A' && hex <= L'F') codepoint += hex - L'A' + 10;
                    else valid = false;
                }
                if (valid) {
                    value.push_back(codepoint);
                    i += 4;
                } else {
                    value.push_back(L'u');
                }
            } else {
                value.push_back(L'u');
            }
            break;
        default: value.push_back(escaped); break;
        }
    }
    return value;
}

std::wstring environment_value(const wchar_t* name, const std::wstring& fallback = {}) {
    const DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
    if (needed == 0) return fallback;
    std::wstring value(static_cast<std::size_t>(needed), L'\0');
    const DWORD actual = GetEnvironmentVariableW(name, value.data(), needed);
    if (actual == 0 || actual >= needed) return fallback;
    value.resize(actual);
    return value;
}

std::wstring expand_environment(const std::wstring& value) {
    const DWORD needed = ExpandEnvironmentStringsW(value.c_str(), nullptr, 0);
    if (needed == 0) return value;
    std::wstring expanded(static_cast<std::size_t>(needed), L'\0');
    const DWORD actual = ExpandEnvironmentStringsW(value.c_str(), expanded.data(), needed);
    if (actual == 0 || actual > needed) return value;
    expanded.resize(actual - 1);
    return expanded;
}

std::wstring registry_string(HKEY key, const wchar_t* name) {
    DWORD type = 0;
    DWORD bytes = 0;
    if (RegQueryValueExW(key, name, nullptr, &type, nullptr, &bytes) != ERROR_SUCCESS ||
        (type != REG_SZ && type != REG_EXPAND_SZ) || bytes < sizeof(wchar_t)) return {};
    std::vector<wchar_t> buffer(bytes / sizeof(wchar_t) + 2, L'\0');
    if (RegQueryValueExW(key, name, nullptr, &type,
        reinterpret_cast<BYTE*>(buffer.data()), &bytes) != ERROR_SUCCESS) return {};
    std::wstring value(buffer.data());
    return type == REG_EXPAND_SZ ? expand_environment(value) : value;
}

std::wstring registry_value(HKEY hive, const wchar_t* subkey,
    const wchar_t* valueName, REGSAM view = 0) {
    HKEY key = nullptr;
    if (RegOpenKeyExW(hive, subkey, 0, KEY_READ | view, &key) != ERROR_SUCCESS) return {};
    const std::wstring value = registry_string(key, valueName);
    RegCloseKey(key);
    return value;
}

std::wstring path_key(const fs::path& path) {
    std::wstring value = path.lexically_normal().wstring();
    replace_all(value, L"/", L"\\");
    return lower(value);
}

bool path_is_inside(const fs::path& path, const fs::path& root) {
    if (path.empty() || root.empty()) return false;
    std::wstring file = path_key(path);
    std::wstring directory = path_key(root);
    if (!directory.empty() && directory.back() != L'\\') directory.push_back(L'\\');
    return file.size() >= directory.size() && file.compare(0, directory.size(), directory) == 0;
}

bool is_file(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path, error);
}

bool is_directory_path(const fs::path& path) {
    std::error_code error;
    return fs::is_directory(path, error);
}

bool game_name_is_blocked(const std::wstring& name) {
    const std::wstring value = lower(name);
    const std::array<const wchar_t*, 18> blocked = {
        L"visual c++", L"redistributable", L".net runtime", L"microsoft edge",
        L"google chrome", L"mozilla firefox", L"steamworks common redistributables",
        L"epic games launcher", L"ubisoft connect", L"ea app", L"gog galaxy",
        L"easy anti-cheat", L"battleye", L"directx", L"vulkan runtime",
        L"nvidia", L"realtek", L"uninstall"
    };
    for (const auto* word : blocked) {
        if (value.find(word) != std::wstring::npos) return true;
    }
    return name.empty();
}

bool executable_is_support_tool(const fs::path& exe) {
    const std::wstring stem = lower(exe.stem().wstring());
    const std::array<const wchar_t*, 15> excluded = {
        L"unins", L"uninstall", L"setup", L"installer", L"redist",
        L"vc_redist", L"dxsetup", L"crash", L"crashreport", L"updater",
        L"patcher", L"easyanticheat", L"battleye", L"support", L"report"
    };
    for (const auto* word : excluded) {
        if (stem.find(word) != std::wstring::npos) return true;
    }
    return false;
}

fs::path resolve_display_icon(std::wstring value) {
    value = trim(std::move(value));
    if (value.empty()) return {};
    if (value.front() == L'"') {
        const auto endQuote = value.find(L'"', 1);
        if (endQuote != std::wstring::npos) value = value.substr(1, endQuote - 1);
    } else {
        const auto ext = lower(value).find(L".exe");
        if (ext != std::wstring::npos) value = value.substr(0, ext + 4);
    }
    fs::path result(expand_environment(value));
    if (is_file(result) && !executable_is_support_tool(result)) return result;
    return {};
}

fs::path find_game_executable(const fs::path& root, const std::wstring& gameName) {
    if (!is_directory_path(root)) return {};
    const std::wstring wanted = normalized_token(gameName);
    fs::path best;
    long long bestScore = std::numeric_limits<long long>::min();
    std::error_code error;
    fs::recursive_directory_iterator it(root,
        fs::directory_options::skip_permission_denied, error), end;
    for (; !error && it != end; it.increment(error)) {
        if (it.depth() >= 2 && it->is_directory(error)) {
            it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(error) || lower(it->path().extension().wstring()) != L".exe") continue;
        const fs::path candidate = it->path();
        if (executable_is_support_tool(candidate)) continue;
        const auto size = fs::file_size(candidate, error);
        if (error) { error.clear(); continue; }
        if (size < 256ULL * 1024ULL) continue;
        const std::wstring stem = normalized_token(candidate.stem().wstring());
        long long score = static_cast<long long>(std::min<std::uintmax_t>(size / (16ULL * 1024ULL * 1024ULL), 20));
        if (!wanted.empty() && stem == wanted) score += 500;
        else if (!wanted.empty() && stem.find(wanted) != std::wstring::npos) score += 300;
        else if (!wanted.empty() && wanted.find(stem) != std::wstring::npos && stem.size() >= 5) score += 200;
        if (score > bestScore) { bestScore = score; best = candidate; }
    }
    return best;
}

std::vector<fs::path> steam_install_roots() {
    std::vector<fs::path> roots;
    const auto add = [&roots](const std::wstring& raw) {
        if (raw.empty()) return;
        fs::path path(expand_environment(raw));
        if (is_directory_path(path) && std::find(roots.begin(), roots.end(), path) == roots.end()) {
            roots.push_back(path);
        }
    };
    add(registry_value(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"SteamPath"));
    add(registry_value(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"InstallPath"));
    add(registry_value(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Valve\\Steam", L"InstallPath", KEY_WOW64_64KEY));
    add(registry_value(HKEY_LOCAL_MACHINE, L"SOFTWARE\\WOW6432Node\\Valve\\Steam", L"InstallPath", KEY_WOW64_64KEY));
    add(registry_value(HKEY_LOCAL_MACHINE, L"SOFTWARE\\WOW6432Node\\Valve\\Steam", L"InstallPath", KEY_WOW64_32KEY));
    const std::wstring programFilesX86 = environment_value(L"ProgramFiles(x86)", L"C:\\Program Files (x86)");
    add(programFilesX86 + L"\\Steam");
    return roots;
}

void add_game(std::vector<GameEntry>& games, GameEntry candidate) {
    if (candidate.name.empty() || candidate.launchTarget.empty() || !is_directory_path(candidate.installPath)) return;
    std::error_code error;
    const fs::path canonical = fs::weakly_canonical(candidate.installPath, error);
    if (!error && !canonical.empty()) candidate.installPath = canonical;
    const std::wstring key = path_key(candidate.installPath);
    for (const auto& game : games) {
        if (path_key(game.installPath) == key) return;
        if (!game.launchIsUri && !candidate.launchIsUri &&
            path_key(fs::path(game.launchTarget)) == path_key(fs::path(candidate.launchTarget))) return;
    }
    games.push_back(std::move(candidate));
}

void scan_steam_games(std::vector<GameEntry>& games) {
    std::vector<fs::path> libraries;
    for (const auto& root : steam_install_roots()) {
        libraries.push_back(root);
        const fs::path config = root / L"steamapps" / L"libraryfolders.vdf";
        const std::wstring text = read_text(config);
        const std::wregex pathExpression(L"\"path\"\\s*\"([^\"]+)\"", std::regex_constants::icase);
        for (auto it = std::wsregex_iterator(text.begin(), text.end(), pathExpression);
            it != std::wsregex_iterator(); ++it) {
            std::wstring libraryPath = (*it)[1].str();
            replace_all(libraryPath, L"\\\\", L"\\");
            fs::path path(libraryPath);
            if (is_directory_path(path) && std::find(libraries.begin(), libraries.end(), path) == libraries.end()) {
                libraries.push_back(path);
            }
        }
    }

    const std::array<const wchar_t*, 8> ignored = {
        L"steamworks common redistributables", L"steam linux runtime",
        L"steam runtime", L"proton", L"steamvr", L"dedicated server",
        L"source sdk", L"server files"
    };
    for (const auto& library : libraries) {
        const fs::path apps = library / L"steamapps";
        if (!is_directory_path(apps)) continue;
        std::error_code error;
        for (fs::directory_iterator it(apps, fs::directory_options::skip_permission_denied, error), end;
            !error && it != end; it.increment(error)) {
            const std::wstring filename = lower(it->path().filename().wstring());
            if (filename.rfind(L"appmanifest_", 0) != 0 || lower(it->path().extension().wstring()) != L".acf") continue;
            const std::wstring text = read_text(it->path());
            const std::wstring appId = extract_vdf_value(text, L"appid");
            const std::wstring name = trim(extract_vdf_value(text, L"name"));
            const std::wstring installDir = extract_vdf_value(text, L"installdir");
            if (appId.empty() || name.empty() || installDir.empty() || game_name_is_blocked(name)) continue;
            bool ignoredGame = false;
            for (const auto* word : ignored) {
                if (lower(name).find(word) != std::wstring::npos) ignoredGame = true;
            }
            if (ignoredGame) continue;
            const fs::path installPath = apps / L"common" / installDir;
            if (!is_directory_path(installPath)) continue;
            add_game(games, GameEntry{
                name, L"Steam", installPath, L"steam://run/" + appId, true
            });
        }
    }
}

void scan_epic_games(std::vector<GameEntry>& games) {
    const fs::path manifests = fs::path(environment_value(L"ProgramData", L"C:\\ProgramData")) /
        L"Epic" / L"EpicGamesLauncher" / L"Data" / L"Manifests";
    if (!is_directory_path(manifests)) return;
    std::error_code error;
    for (fs::directory_iterator it(manifests, fs::directory_options::skip_permission_denied, error), end;
        !error && it != end; it.increment(error)) {
        if (lower(it->path().extension().wstring()) != L".item") continue;
        const std::wstring text = read_text(it->path());
        const std::wstring name = trim(extract_json_value(text, L"DisplayName"));
        const std::wstring install = extract_json_value(text, L"InstallLocation");
        const std::wstring appName = extract_json_value(text, L"AppName");
        if (name.empty() || install.empty() || appName.empty() || game_name_is_blocked(name)) continue;
        const fs::path root(install);
        if (!is_directory_path(root)) continue;
        const std::wstring uri = L"com.epicgames.launcher://apps/" + appName + L"?action=launch&silent=true";
        add_game(games, GameEntry{name, L"Epic Games", root, uri, true});
    }
}

void scan_common_library_folder(std::vector<GameEntry>& games,
    const fs::path& libraryRoot, const std::wstring& source) {
    if (!is_directory_path(libraryRoot)) return;
    std::error_code error;
    for (fs::directory_iterator it(libraryRoot, fs::directory_options::skip_permission_denied, error), end;
        !error && it != end; it.increment(error)) {
        if (!it->is_directory(error)) continue;
        const fs::path root = it->path();
        const std::wstring name = root.filename().wstring();
        if (game_name_is_blocked(name)) continue;
        const fs::path exe = find_game_executable(root, name);
        if (exe.empty()) continue;
        add_game(games, GameEntry{name, source, root, exe.wstring(), false});
    }
}

void scan_registry_games(std::vector<GameEntry>& games) {
    const wchar_t* uninstallKey = L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall";
    const std::array<HKEY, 2> hives = {HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    const std::array<REGSAM, 2> views = {KEY_WOW64_64KEY, KEY_WOW64_32KEY};
    for (HKEY hive : hives) {
        for (REGSAM view : views) {
            HKEY root = nullptr;
            if (RegOpenKeyExW(hive, uninstallKey, 0, KEY_READ | view, &root) != ERROR_SUCCESS) continue;
            for (DWORD index = 0;; ++index) {
                wchar_t subname[512]{};
                DWORD subLength = static_cast<DWORD>(std::size(subname));
                const LONG result = RegEnumKeyExW(root, index, subname, &subLength, nullptr, nullptr, nullptr, nullptr);
                if (result != ERROR_SUCCESS) break;
                HKEY item = nullptr;
                if (RegOpenKeyExW(root, subname, 0, KEY_READ, &item) != ERROR_SUCCESS) continue;
                const std::wstring name = trim(registry_string(item, L"DisplayName"));
                const std::wstring installValue = registry_string(item, L"InstallLocation");
                const std::wstring displayIcon = registry_string(item, L"DisplayIcon");
                RegCloseKey(item);
                if (name.empty() || game_name_is_blocked(name)) continue;
                const fs::path icon = resolve_display_icon(displayIcon);
                fs::path installPath = expand_environment(installValue);
                if (!is_directory_path(installPath) && !icon.empty()) installPath = icon.parent_path();
                if (!is_directory_path(installPath)) continue;
                fs::path exe = icon;
                if (exe.empty()) exe = find_game_executable(installPath, name);
                if (exe.empty() || !is_file(exe) || executable_is_support_tool(exe)) continue;
                std::error_code error;
                if (fs::file_size(exe, error) < 256ULL * 1024ULL || error) continue;
                add_game(games, GameEntry{name, L"Installed game", installPath, exe.wstring(), false});
            }
            RegCloseKey(root);
        }
    }
}

void scan_library(std::vector<GameEntry>& games) {
    games.clear();
    scan_steam_games(games);
    scan_epic_games(games);

    const auto programFiles = fs::path(environment_value(L"ProgramFiles", L"C:\\Program Files"));
    const auto programFilesX86 = fs::path(environment_value(L"ProgramFiles(x86)", L"C:\\Program Files (x86)"));
    const auto programData = fs::path(environment_value(L"ProgramData", L"C:\\ProgramData"));
    const auto userProfile = fs::path(environment_value(L"USERPROFILE", L""));
    scan_common_library_folder(games, programFilesX86 / L"Ubisoft" / L"Ubisoft Game Launcher" / L"games", L"Ubisoft Connect");
    scan_common_library_folder(games, programFiles / L"Ubisoft" / L"Ubisoft Game Launcher" / L"games", L"Ubisoft Connect");
    scan_common_library_folder(games, programFiles / L"EA Games", L"EA app");
    scan_common_library_folder(games, programFilesX86 / L"Origin Games", L"EA app");
    scan_common_library_folder(games, programFiles / L"GOG Galaxy" / L"Games", L"GOG Galaxy");
    scan_common_library_folder(games, fs::path(L"C:\\GOG Games"), L"GOG");
    scan_common_library_folder(games, fs::path(L"C:\\XboxGames"), L"Xbox");
    scan_common_library_folder(games, fs::path(L"C:\\Games"), L"Local game folder");
    if (!userProfile.empty()) {
        scan_common_library_folder(games, userProfile / L"Games", L"Local game folder");
        scan_common_library_folder(games, userProfile / L"GOG Games", L"GOG");
    }
    scan_registry_games(games);
    std::sort(games.begin(), games.end(), [](const GameEntry& a, const GameEntry& b) {
        return lower(a.name) < lower(b.name);
    });
}

std::wstring current_utc() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_s(&tm, &raw);
    std::wostringstream out;
    out << std::put_time(&tm, L"%Y-%m-%dT%H:%M:%S") << L'.'
        << std::setw(3) << std::setfill(L'0') << milliseconds.count() << L'Z';
    return out.str();
}

std::wstring compact_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_s(&tm, &raw);
    std::wostringstream out;
    out << std::put_time(&tm, L"%Y%m%dT%H%M%SZ");
    return out.str();
}

std::wstring safe_slug(std::wstring value) {
    value = lower(std::move(value));
    for (auto& ch : value) {
        if (!std::iswalnum(ch) && ch != L'-' && ch != L'_') ch = L'-';
    }
    while (!value.empty() && value.front() == L'-') value.erase(value.begin());
    while (!value.empty() && value.back() == L'-') value.pop_back();
    if (value.empty()) value = L"game";
    if (value.size() > 64) value.resize(64);
    return value;
}

std::wstring base64_encode(const std::vector<unsigned char>& bytes) {
    static constexpr wchar_t alphabet[] =
        L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::wstring output;
    output.reserve(((bytes.size() + 2) / 3) * 4);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t a = bytes[i];
        const std::uint32_t b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
        const std::uint32_t c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
        const std::uint32_t value = (a << 16) | (b << 8) | c;
        output.push_back(alphabet[(value >> 18) & 0x3f]);
        output.push_back(alphabet[(value >> 12) & 0x3f]);
        output.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 0x3f] : L'=');
        output.push_back(i + 2 < bytes.size() ? alphabet[value & 0x3f] : L'=');
    }
    return output;
}

bool jpeg_encoder_clsid(CLSID& clsid) {
    return SUCCEEDED(CLSIDFromString(
        L"{557CF401-1A04-11D3-9A73-0000F81EF32E}", &clsid));
}

std::vector<unsigned char> encode_jpeg(Bitmap& bitmap) {
    std::vector<unsigned char> bytes;
    IStream* stream = nullptr;
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) return bytes;
    CLSID jpeg{};
    if (!jpeg_encoder_clsid(jpeg)) {
        stream->Release();
        return bytes;
    }
    ULONG quality = 68;
    EncoderParameters parameters{};
    parameters.Count = 1;
    parameters.Parameter[0].Guid = EncoderQuality;
    parameters.Parameter[0].Type = EncoderParameterValueTypeLong;
    parameters.Parameter[0].NumberOfValues = 1;
    parameters.Parameter[0].Value = &quality;
    if (bitmap.Save(stream, &jpeg, &parameters) == Ok) {
        STATSTG status{};
        if (SUCCEEDED(stream->Stat(&status, STATFLAG_NONAME)) &&
            status.cbSize.QuadPart > 0 &&
            status.cbSize.QuadPart <= std::numeric_limits<std::size_t>::max()) {
            HGLOBAL memory = nullptr;
            if (SUCCEEDED(GetHGlobalFromStream(stream, &memory))) {
                const SIZE_T size = static_cast<SIZE_T>(status.cbSize.QuadPart);
                void* data = GlobalLock(memory);
                if (data != nullptr) {
                    const auto* begin = static_cast<const unsigned char*>(data);
                    bytes.assign(begin, begin + size);
                    GlobalUnlock(memory);
                }
            }
        }
    }
    stream->Release();
    return bytes;
}

CapturedImage capture_window(HWND window) {
    CapturedImage result;
    if (!IsWindow(window) || IsIconic(window) || !IsWindowVisible(window)) return result;
    RECT bounds{};
    if (!GetWindowRect(window, &bounds)) return result;
    const int width = bounds.right - bounds.left;
    const int height = bounds.bottom - bounds.top;
    if (width < 320 || height < 200 || width > 16384 || height > 16384) return result;

    HDC screen = GetDC(nullptr);
    if (screen == nullptr) return result;
    HDC memory = CreateCompatibleDC(screen);
    if (memory == nullptr) {
        ReleaseDC(nullptr, screen);
        return result;
    }
    HBITMAP bitmapHandle = CreateCompatibleBitmap(screen, width, height);
    if (bitmapHandle == nullptr) {
        DeleteDC(memory);
        ReleaseDC(nullptr, screen);
        return result;
    }
    HGDIOBJ oldBitmap = SelectObject(memory, bitmapHandle);
    const BOOL copied = BitBlt(memory, 0, 0, width, height, screen,
        bounds.left, bounds.top, SRCCOPY | CAPTUREBLT);
    SelectObject(memory, oldBitmap);
    DeleteDC(memory);
    ReleaseDC(nullptr, screen);
    if (!copied) {
        DeleteObject(bitmapHandle);
        return result;
    }

    Bitmap* original = Bitmap::FromHBITMAP(bitmapHandle, nullptr);
    DeleteObject(bitmapHandle);
    if (original == nullptr || original->GetLastStatus() != Ok) {
        delete original;
        return result;
    }

    const double scale = std::min(1.0, static_cast<double>(kMaximumImageDimension) /
        static_cast<double>(std::max(width, height)));
    const int outputWidth = std::max(1, static_cast<int>(std::lround(width * scale)));
    const int outputHeight = std::max(1, static_cast<int>(std::lround(height * scale)));
    result.sourceWidth = width;
    result.sourceHeight = height;
    result.width = outputWidth;
    result.height = outputHeight;
    if (outputWidth == width && outputHeight == height) {
        result.jpeg = encode_jpeg(*original);
    } else {
        Bitmap resized(outputWidth, outputHeight, PixelFormat24bppRGB);
        Graphics graphics(&resized);
        graphics.SetInterpolationMode(InterpolationModeHighQualityBicubic);
        graphics.SetCompositingQuality(CompositingQualityHighQuality);
        if (graphics.DrawImage(original, Rect(0, 0, outputWidth, outputHeight),
            0, 0, width, height, UnitPixel) == Ok) {
            result.jpeg = encode_jpeg(resized);
        }
    }
    delete original;
    return result;
}

std::uint64_t filetime_value(const FILETIME& value) {
    ULARGE_INTEGER integer{};
    integer.LowPart = value.dwLowDateTime;
    integer.HighPart = value.dwHighDateTime;
    return integer.QuadPart;
}

FrameTelemetry sample_telemetry(DWORD pid) {
    FrameTelemetry telemetry;
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (process == nullptr) process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (process == nullptr) return telemetry;

    PROCESS_MEMORY_COUNTERS_EX memory{};
    memory.cb = sizeof(memory);
    if (GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
        sizeof(memory))) {
        telemetry.hasWorkingSet = true;
        telemetry.workingSetBytes = static_cast<std::uint64_t>(memory.WorkingSetSize);
    }

    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(process, &created, &exited, &kernel, &user)) {
        const std::uint64_t totalCpu = filetime_value(kernel) + filetime_value(user);
        static DWORD previousPid = 0;
        static std::uint64_t previousCpu = 0;
        static Clock::time_point previousAt{};
        const auto now = Clock::now();
        if (previousPid == pid && previousAt.time_since_epoch().count() != 0) {
            const auto elapsed = std::chrono::duration<double>(now - previousAt).count();
            if (elapsed > 0.0 && totalCpu >= previousCpu) {
                const double cpuSeconds = static_cast<double>(totalCpu - previousCpu) / 10000000.0;
                const DWORD cores = std::max<DWORD>(1, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
                telemetry.cpuPercent = std::clamp((cpuSeconds / elapsed) * 100.0 / cores, 0.0, 100.0);
                telemetry.hasCpu = true;
            }
        }
        previousPid = pid;
        previousCpu = totalCpu;
        previousAt = now;
    }
    CloseHandle(process);
    return telemetry;
}

class JsonChunkWriter {
public:
    void configure(const fs::path& outputRoot, const GameEntry& game,
        DWORD pid, const std::wstring& sessionId, const std::wstring& executableName) {
        close_chunk();
        outputRoot_ = outputRoot;
        gameName_ = game.name;
        source_ = game.source;
        pid_ = pid;
        sessionId_ = sessionId;
        executableName_ = executableName;
        sessionDirectory_.clear();
        chunkIndex_ = 0;
        frameCountInChunk_ = 0;
        totalFrames_ = 0;
        lastError_.clear();
    }

    bool append(std::uint64_t frameIndex, const std::wstring& capturedUtc,
        std::uint64_t elapsedMs, const CapturedImage& image, const FrameTelemetry& telemetry) {
        if (image.jpeg.empty()) {
            lastError_ = L"Windows could not capture this game window.";
            return false;
        }
        if (!ensure_session_directory()) return false;
        const std::string json = make_frame_json(frameIndex, capturedUtc, elapsedMs, image, telemetry);
        if (!stream_.is_open() && !open_next_chunk()) return false;
        const std::uint64_t commaBytes = frameCountInChunk_ == 0 ? 0 : 1;
        const std::uint64_t closingBytes = 3; // ]}, followed by a newline
        if (currentBytes_ + commaBytes + json.size() + closingBytes > kMaximumJsonBytes) {
            close_chunk();
            if (!open_next_chunk()) return false;
        }
        const std::uint64_t nextCommaBytes = frameCountInChunk_ == 0 ? 0 : 1;
        if (currentBytes_ + nextCommaBytes + json.size() + closingBytes > kMaximumJsonBytes) {
            lastError_ = L"A single captured frame exceeds the 23 MiB JSON limit.";
            close_chunk();
            return false;
        }
        if (frameCountInChunk_ > 0) stream_.put(',');
        stream_.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!stream_) {
            lastError_ = L"Could not write the sample JSON file. Check free disk space and folder permissions.";
            close_chunk();
            return false;
        }
        currentBytes_ += nextCommaBytes + json.size();
        ++frameCountInChunk_;
        ++totalFrames_;
        stream_.flush();
        return static_cast<bool>(stream_);
    }

    void close_chunk() {
        if (!stream_.is_open()) return;
        if (frameCountInChunk_ == 0) {
            stream_.close();
            std::error_code error;
            fs::remove(currentPartPath_, error);
            currentPartPath_.clear();
            currentFinalPath_.clear();
            currentBytes_ = 0;
            return;
        }
        stream_.write("]}\n", 3);
        currentBytes_ += 3;
        stream_.flush();
        stream_.close();
        std::error_code error;
        fs::rename(currentPartPath_, currentFinalPath_, error);
        if (error) {
            lastError_ = L"The JSON chunk was closed, but Windows could not finalize its filename.";
        }
        currentPartPath_.clear();
        currentFinalPath_.clear();
        currentBytes_ = 0;
        frameCountInChunk_ = 0;
    }

    void close() {
        close_chunk();
        sessionDirectory_.clear();
        chunkIndex_ = 0;
        totalFrames_ = 0;
    }

    std::uint64_t total_frames() const { return totalFrames_; }
    int chunk_index() const { return chunkIndex_; }
    std::wstring last_error() const { return lastError_; }
    fs::path last_directory() const { return sessionDirectory_; }
    fs::path output_root() const { return outputRoot_; }

private:
    bool ensure_session_directory() {
        if (!sessionDirectory_.empty()) return true;
        if (outputRoot_.empty() || sessionId_.empty()) {
            lastError_ = L"The sample output folder is not configured.";
            return false;
        }
        sessionDirectory_ = outputRoot_ / safe_slug(gameName_) / sessionId_;
        std::error_code error;
        fs::create_directories(sessionDirectory_, error);
        if (error) {
            lastError_ = L"Could not create the sample folder. Choose a writable output location.";
            sessionDirectory_.clear();
            return false;
        }
        return true;
    }

    bool open_next_chunk() {
        if (!ensure_session_directory()) return false;
        ++chunkIndex_;
        frameCountInChunk_ = 0;
        currentBytes_ = 0;
        std::wostringstream filename;
        filename << L"chunk-" << std::setw(4) << std::setfill(L'0') << chunkIndex_ << L".json";
        currentFinalPath_ = sessionDirectory_ / filename.str();
        currentPartPath_ = currentFinalPath_;
        currentPartPath_ += L".part";
        std::error_code error;
        if (fs::exists(currentFinalPath_, error) || fs::exists(currentPartPath_, error)) {
            lastError_ = L"A sample chunk with this name already exists; start a new session.";
            return false;
        }
        stream_.open(currentPartPath_, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!stream_) {
            lastError_ = L"Could not open a sample JSON file for writing.";
            return false;
        }
        const std::string header =
            "{\n\"schema\":\"trss.sample-recorder.chunk.v1\","
            "\"sessionId\":\"" + json_escape(to_utf8(sessionId_)) + "\","
            "\"chunkIndex\":" + std::to_string(chunkIndex_) + ","
            "\"game\":{\"name\":\"" + json_escape(to_utf8(gameName_)) + "\","
            "\"source\":\"" + json_escape(to_utf8(source_)) + "\","
            "\"executableName\":\"" + json_escape(to_utf8(executableName_)) + "\"},"
            "\"processId\":" + std::to_string(pid_) + ","
            "\"capture\":{\"warmupSeconds\":120,\"sampleIntervalMs\":500,"
            "\"maxJsonBytes\":24117248,\"imageFormat\":\"image/jpeg\","
            "\"imageEncoding\":\"base64\",\"pixelSource\":\"visible-foreground-window\","
            "\"maximumImageDimension\":1920,\"auxiliaryBuffers\":false},"
            "\"limits\":[\"RGB color pixels only; no depth or motion vectors\","
            "\"No game injection or private renderer access\","
            "\"Samples are collected only while this game window is foreground\"],"
            "\"frames\":[";
        if (header.size() + 3 > kMaximumJsonBytes) {
            lastError_ = L"The JSON header exceeds the file size limit.";
            stream_.close();
            fs::remove(currentPartPath_, error);
            return false;
        }
        stream_.write(header.data(), static_cast<std::streamsize>(header.size()));
        currentBytes_ = header.size();
        return static_cast<bool>(stream_);
    }

    std::string make_frame_json(std::uint64_t frameIndex, const std::wstring& capturedUtc,
        std::uint64_t elapsedMs, const CapturedImage& image, const FrameTelemetry& telemetry) {
        std::ostringstream json;
        json << "{\"frameIndex\":" << frameIndex
            << ",\"capturedAtUtc\":\"" << json_escape(to_utf8(capturedUtc)) << "\""
            << ",\"elapsedMs\":" << elapsedMs
            << ",\"sourceWidth\":" << image.sourceWidth
            << ",\"sourceHeight\":" << image.sourceHeight
            << ",\"width\":" << image.width
            << ",\"height\":" << image.height
            << ",\"downscaledTo1920\":"
            << ((image.sourceWidth > kMaximumImageDimension || image.sourceHeight > kMaximumImageDimension) ? "true" : "false")
            << ",\"format\":\"image/jpeg\""
            << ",\"imageBase64\":\"" << to_utf8(base64_encode(image.jpeg)) << "\""
            << ",\"processCpuPercent\":";
        if (telemetry.hasCpu) json << std::fixed << std::setprecision(2) << telemetry.cpuPercent;
        else json << "null";
        json << ",\"workingSetBytes\":";
        if (telemetry.hasWorkingSet) json << telemetry.workingSetBytes;
        else json << "null";
        json << '}';
        return json.str();
    }

    fs::path outputRoot_;
    fs::path sessionDirectory_;
    fs::path currentPartPath_;
    fs::path currentFinalPath_;
    std::wofstream unusedStream_;
    std::ofstream stream_;
    std::wstring gameName_;
    std::wstring source_;
    std::wstring sessionId_;
    std::wstring executableName_;
    DWORD pid_ = 0;
    int chunkIndex_ = 0;
    std::uint64_t currentBytes_ = 0;
    std::uint64_t frameCountInChunk_ = 0;
    std::uint64_t totalFrames_ = 0;
    std::wstring lastError_;
};

JsonChunkWriter g_writer;

std::wstring documents_folder() {
    PWSTR rawPath = nullptr;
    std::wstring result;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, KF_FLAG_DEFAULT, nullptr, &rawPath)) &&
        rawPath != nullptr) {
        result = rawPath;
        CoTaskMemFree(rawPath);
    }
    if (result.empty()) result = environment_value(L"USERPROFILE", L"C:\\");
    return result;
}

bool choose_output_folder(fs::path& chosen) {
    BROWSEINFOW browser{};
    browser.hwndOwner = g_window;
    browser.lpszTitle = L"Choose where TrSS Sample Recorder saves JSON chunks";
    browser.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX;
    PIDLIST_ABSOLUTE item = SHBrowseForFolderW(&browser);
    if (item == nullptr) return false;
    wchar_t path[MAX_PATH]{};
    const bool ok = SHGetPathFromIDListW(item, path) != FALSE;
    CoTaskMemFree(item);
    if (!ok || path[0] == L'\0') return false;
    chosen = path;
    return true;
}

void set_text_if_changed(HWND control, std::wstring& cache, const std::wstring& value) {
    if (control == nullptr || cache == value) return;
    cache = value;
    SetWindowTextW(control, value.c_str());
}

void set_status(const std::wstring& value) {
    set_text_if_changed(g_status, g_statusCache, value);
}

void set_active_game_text(const std::wstring& value) {
    set_text_if_changed(g_activeGame, g_activeCache, value);
}

void set_details_text(const std::wstring& value) {
    set_text_if_changed(g_details, g_detailsCache, value);
}

void set_output_text() {
    const std::wstring value = L"Saving JSON samples to: " + g_outputRoot.wstring();
    set_text_if_changed(g_outputText, g_outputCache, value);
}

int selected_game_index() {
    const LRESULT selected = SendMessageW(g_gameList, LB_GETCURSEL, 0, 0);
    if (selected == LB_ERR) return -1;
    const LRESULT index = SendMessageW(g_gameList, LB_GETITEMDATA, selected, 0);
    if (index == LB_ERR || index < 0 || static_cast<std::size_t>(index) >= g_games.size()) return -1;
    return static_cast<int>(index);
}

void update_game_details() {
    const int index = selected_game_index();
    if (index < 0) {
        set_details_text(L"Select a game to see its detected install location and launch source.");
        return;
    }
    const auto& game = g_games[static_cast<std::size_t>(index)];
    set_details_text(L"Name: " + game.name + L"\r\nStore: " + game.source +
        L"\r\nInstall location: " + game.installPath.wstring() +
        L"\r\nLaunch method: " + (game.launchIsUri ? L"Game launcher" : L"Installed executable"));
}

void populate_game_list() {
    const int previous = selected_game_index();
    std::wstring previousName;
    if (previous >= 0) previousName = g_games[static_cast<std::size_t>(previous)].name;
    SendMessageW(g_gameList, LB_RESETCONTENT, 0, 0);
    for (std::size_t i = 0; i < g_games.size(); ++i) {
        const auto line = g_games[i].name + L"    [" + g_games[i].source + L"]";
        const LRESULT item = SendMessageW(g_gameList, LB_ADDSTRING, 0,
            reinterpret_cast<LPARAM>(line.c_str()));
        if (item != LB_ERR && item != LB_ERRSPACE) {
            SendMessageW(g_gameList, LB_SETITEMDATA, item, static_cast<LPARAM>(i));
        }
    }
    if (!g_games.empty()) {
        LRESULT wanted = 0;
        if (!previousName.empty()) {
            for (std::size_t i = 0; i < g_games.size(); ++i) {
                if (g_games[i].name == previousName) { wanted = static_cast<LRESULT>(i); break; }
            }
        }
        SendMessageW(g_gameList, LB_SETCURSEL, wanted, 0);
    }
    update_game_details();
}

void stop_tracking();

void do_scan_library() {
    std::wstring trackedRoot;
    if (g_trackedPid != 0 && g_trackedGameIndex >= 0 &&
        static_cast<std::size_t>(g_trackedGameIndex) < g_games.size()) {
        trackedRoot = path_key(g_games[static_cast<std::size_t>(g_trackedGameIndex)].installPath);
    }
    scan_library(g_games);
    if (!trackedRoot.empty()) {
        const auto existing = std::find_if(g_games.begin(), g_games.end(),
            [&trackedRoot](const GameEntry& game) {
                return path_key(game.installPath) == trackedRoot;
            });
        if (existing == g_games.end()) {
            stop_tracking();
        } else {
            g_trackedGameIndex = static_cast<int>(std::distance(g_games.begin(), existing));
            g_trackingDetailsDirty = true;
        }
    }
    populate_game_list();
    if (g_games.empty()) {
        set_status(L"No compatible installed games were found. Steam and Epic libraries, common PC game folders, and Windows game registry entries were checked.");
    } else {
        set_status(L"Found " + std::to_wstring(g_games.size()) +
            L" installed games. Running games are detected automatically; capture starts after a 120-second warm-up.");
    }
}

bool launch_selected_game() {
    const int index = selected_game_index();
    if (index < 0) {
        MessageBoxW(g_window, L"Select a game from the library first.", L"TrSS Sample Recorder", MB_OK | MB_ICONINFORMATION);
        return false;
    }
    const auto& game = g_games[static_cast<std::size_t>(index)];
    const wchar_t* verb = L"open";
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(g_window, verb,
        game.launchTarget.c_str(), nullptr, game.installPath.c_str(), SW_SHOWNORMAL));
    if (result <= 32) {
        std::wstring message = L"Windows could not launch this game through the detected entry.\r\n\r\n"
            L"Game: " + game.name + L"\r\nLocation: " + game.installPath.wstring() +
            L"\r\n\r\nTry launching it from its original launcher once. TrSS will still detect it automatically.";
        MessageBoxW(g_window, message.c_str(), L"Could not launch game", MB_OK | MB_ICONWARNING);
        return false;
    }
    set_status(L"Launching " + game.name +
        L". The 120-second warm-up begins when its game window is detected.");
    return true;
}

std::wstring get_process_image_path(DWORD pid) {
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (process == nullptr) return {};
    std::vector<wchar_t> buffer(32768, L'\0');
    DWORD size = static_cast<DWORD>(buffer.size());
    std::wstring result;
    if (QueryFullProcessImageNameW(process, 0, buffer.data(), &size)) {
        result.assign(buffer.data(), size);
    }
    CloseHandle(process);
    return result;
}

int match_game_path(const std::wstring& executablePath) {
    if (executablePath.empty()) return -1;
    const fs::path path(executablePath);
    for (std::size_t i = 0; i < g_games.size(); ++i) {
        if (path_is_inside(path, g_games[i].installPath)) return static_cast<int>(i);
    }
    return -1;
}

BOOL CALLBACK enumerate_game_windows(HWND window, LPARAM parameter) {
    auto* candidates = reinterpret_cast<std::vector<WindowCandidate>*>(parameter);
    if (!IsWindowVisible(window) || IsIconic(window) || GetWindow(window, GW_OWNER) != nullptr) return TRUE;
    RECT bounds{};
    if (!GetWindowRect(window, &bounds)) return TRUE;
    const int width = bounds.right - bounds.left;
    const int height = bounds.bottom - bounds.top;
    if (width < 480 || height < 270) return TRUE;
    DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    if (pid == 0 || pid == GetCurrentProcessId()) return TRUE;
    const std::wstring path = get_process_image_path(pid);
    const int gameIndex = match_game_path(path);
    if (gameIndex < 0) return TRUE;
    candidates->push_back(WindowCandidate{window, pid, gameIndex, path});
    return TRUE;
}

std::wstring make_session_id(DWORD pid) {
    return compact_utc() + L"-" + std::to_wstring(pid) + L"-" +
        std::to_wstring(GetTickCount64() % 1000000ULL);
}

void stop_tracking() {
    g_writer.close();
    g_trackedPid = 0;
    g_trackedGameIndex = -1;
    g_trackedWindow = nullptr;
    g_trackedExecutable.clear();
    g_sessionId.clear();
    g_frameIndex = 0;
    g_captureErrorShown = false;
    g_trackingDetailsDirty = true;
}

void start_tracking(const WindowCandidate& candidate) {
    g_writer.close();
    g_trackedPid = candidate.processId;
    g_trackedGameIndex = candidate.gameIndex;
    g_trackedWindow = candidate.window;
    g_trackedExecutable = fs::path(candidate.executablePath).filename().wstring();
    g_trackedAt = Clock::now();
    g_lastCapture = g_trackedAt - std::chrono::milliseconds(kSampleIntervalMs);
    g_sessionId = make_session_id(candidate.processId);
    g_frameIndex = 0;
    g_captureErrorShown = false;
    g_writer.configure(g_outputRoot, g_games[static_cast<std::size_t>(candidate.gameIndex)],
        candidate.processId, g_sessionId, g_trackedExecutable);
    g_trackingDetailsDirty = true;
}

bool tracked_process_is_alive() {
    if (g_trackedPid == 0 || g_trackedGameIndex < 0 ||
        static_cast<std::size_t>(g_trackedGameIndex) >= g_games.size()) return false;
    const auto path = get_process_image_path(g_trackedPid);
    return !path.empty() && path_is_inside(fs::path(path),
        g_games[static_cast<std::size_t>(g_trackedGameIndex)].installPath);
}

void poll_game_detection() {
    std::vector<WindowCandidate> candidates;
    EnumWindows(enumerate_game_windows, reinterpret_cast<LPARAM>(&candidates));
    const HWND foreground = GetForegroundWindow();
    DWORD foregroundPid = 0;
    if (foreground != nullptr) GetWindowThreadProcessId(foreground, &foregroundPid);

    auto selected = candidates.end();
    if (foregroundPid != 0) {
        selected = std::find_if(candidates.begin(), candidates.end(),
            [foregroundPid](const WindowCandidate& c) { return c.processId == foregroundPid; });
    }
    if (selected == candidates.end() && g_trackedPid != 0) {
        selected = std::find_if(candidates.begin(), candidates.end(),
            [](const WindowCandidate& c) { return c.processId == g_trackedPid; });
    }
    if (selected == candidates.end() && !candidates.empty()) selected = candidates.begin();

    if (selected != candidates.end()) {
        if (selected->processId != g_trackedPid || selected->gameIndex != g_trackedGameIndex) {
            start_tracking(*selected);
        } else {
            g_trackedWindow = selected->window;
            g_trackedExecutable = fs::path(selected->executablePath).filename().wstring();
        }
        return;
    }

    if (g_trackedPid != 0 && tracked_process_is_alive()) return;
    if (g_trackedPid != 0) stop_tracking();
}

bool capture_current_frame() {
    if (g_trackedPid == 0 || g_trackedGameIndex < 0 ||
        static_cast<std::size_t>(g_trackedGameIndex) >= g_games.size()) return false;
    const HWND foreground = GetForegroundWindow();
    if (foreground == nullptr) return false;
    DWORD foregroundPid = 0;
    GetWindowThreadProcessId(foreground, &foregroundPid);
    if (foregroundPid != g_trackedPid) return false;

    CapturedImage image = capture_window(foreground);
    if (image.jpeg.empty()) {
        if (!g_captureErrorShown) {
            set_status(L"The game is detected, but Windows could not read its visible pixels. Try Borderless Windowed mode and bring the game to the foreground.");
            g_captureErrorShown = true;
        }
        return false;
    }
    const FrameTelemetry telemetry = sample_telemetry(g_trackedPid);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - g_trackedAt).count();
    const bool wrote = g_writer.append(g_frameIndex, current_utc(),
        elapsed < 0 ? 0 : static_cast<std::uint64_t>(elapsed), image, telemetry);
    if (!wrote) {
        if (!g_writer.last_error().empty()) set_status(g_writer.last_error());
        return false;
    }
    ++g_frameIndex;
    g_captureErrorShown = false;
    return true;
}

std::wstring remaining_text(long long milliseconds) {
    const long long seconds = std::max<long long>(0, (milliseconds + 999) / 1000);
    std::wostringstream out;
    out << std::setw(2) << std::setfill(L'0') << (seconds / 60) << L':'
        << std::setw(2) << std::setfill(L'0') << (seconds % 60);
    return out.str();
}

void update_monitoring_state() {
    const auto now = Clock::now();
    if (g_lastDetection.time_since_epoch().count() == 0 ||
        now - g_lastDetection >= std::chrono::seconds(1)) {
        g_lastDetection = now;
        poll_game_detection();
    }

    if (g_trackedPid == 0 || g_trackedGameIndex < 0 ||
        static_cast<std::size_t>(g_trackedGameIndex) >= g_games.size()) {
        set_active_game_text(L"No running game detected");
        set_status(L"Monitoring installed-game windows. Launch a game from this library or start it normally; recording is automatic after the warm-up.");
        SendMessageW(g_progress, PBM_SETPOS, 0, 0);
        if (g_trackingDetailsDirty) {
            update_game_details();
            g_trackingDetailsDirty = false;
        }
        return;
    }

    const GameEntry& game = g_games[static_cast<std::size_t>(g_trackedGameIndex)];
    set_active_game_text(L"Detected: " + game.name);
    if (g_trackingDetailsDirty) {
        std::wstring detail = L"Store: " + game.source +
            L"\r\nProcess: " + g_trackedExecutable +
            L"\r\nPID: " + std::to_wstring(g_trackedPid) +
            L"\r\nInstall location: " + game.installPath.wstring() +
            L"\r\n\r\nWarm-up: 120 seconds after the game window is detected.\r\n"
            L"Capture is limited to the foreground game window. Samples stay on this PC.";
        set_details_text(detail);
        g_trackingDetailsDirty = false;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - g_trackedAt).count();
    const auto warmupMs = static_cast<long long>(kWarmupSeconds) * 1000;
    if (elapsed < warmupMs) {
        const long long left = warmupMs - elapsed;
        SendMessageW(g_progress, PBM_SETPOS,
            static_cast<WPARAM>(std::clamp<long long>((elapsed * 100) / warmupMs, 0, 100)), 0);
        set_status(L"Warm-up in progress. Automatic capture starts in " +
            remaining_text(left) + L". You can keep playing; no frames are saved yet.");
        return;
    }
    SendMessageW(g_progress, PBM_SETPOS, 100, 0);
    if (SendMessageW(g_autoCapture, BM_GETCHECK, 0, 0) != BST_CHECKED) {
        g_writer.close_chunk();
        set_status(L"Automatic capture is paused. The game is still detected; enable the checkbox to resume sampling.");
        return;
    }

    const HWND foreground = GetForegroundWindow();
    DWORD foregroundPid = 0;
    if (foreground != nullptr) GetWindowThreadProcessId(foreground, &foregroundPid);
    if (foregroundPid != g_trackedPid) {
        g_writer.close_chunk();
        set_status(L"Ready to sample " + game.name +
            L". Bring its window to the foreground; other desktop windows are never recorded.");
        return;
    }
    if (now - g_lastCapture >= std::chrono::milliseconds(kSampleIntervalMs)) {
        g_lastCapture = now;
        capture_current_frame();
    }
    if (g_writer.total_frames() > 0) {
        set_status(L"Recording " + game.name + L" · " +
            std::to_wstring(g_writer.total_frames()) + L" frames captured · " +
            std::to_wstring(g_writer.chunk_index()) +
            L" JSON chunk(s) in this session. Every completed file is capped at 23 MiB.");
    } else if (g_statusCache.find(L"could not read its visible pixels") == std::wstring::npos) {
        set_status(L"Capture is armed for " + game.name +
            L". Visible frames will be saved to JSON when the game is in front.");
    }
}

void on_choose_folder() {
    fs::path chosen;
    if (!choose_output_folder(chosen)) return;
    g_writer.close();
    g_outputRoot = chosen;
    if (g_trackedPid != 0 && g_trackedGameIndex >= 0 &&
        static_cast<std::size_t>(g_trackedGameIndex) < g_games.size()) {
        g_sessionId = make_session_id(g_trackedPid);
        g_frameIndex = 0;
        g_writer.configure(g_outputRoot, g_games[static_cast<std::size_t>(g_trackedGameIndex)],
            g_trackedPid, g_sessionId, g_trackedExecutable);
    }
    set_output_text();
    set_status(L"Sample output folder changed. Completed JSON files will be saved to the selected location.");
}

void on_open_folder() {
    std::error_code error;
    fs::create_directories(g_outputRoot, error);
    if (error) {
        MessageBoxW(g_window, L"Could not open the samples folder. Choose another writable folder.",
            L"TrSS Sample Recorder", MB_OK | MB_ICONWARNING);
        return;
    }
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(
        g_window, L"open", g_outputRoot.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (result <= 32) {
        MessageBoxW(g_window, L"Windows could not open the samples folder.",
            L"TrSS Sample Recorder", MB_OK | MB_ICONWARNING);
    }
}

void create_child(HWND parent, HWND& destination, const wchar_t* className,
    const wchar_t* text, DWORD style, int id, HFONT font = nullptr) {
    destination = CreateWindowExW(0, className, text,
        WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10, parent,
        id == 0 ? nullptr : reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        g_instance, nullptr);
    if (destination != nullptr && font != nullptr) {
        SendMessageW(destination, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    }
}

void layout_controls(int width, int height) {
    const int margin = 24;
    const int panelTop = 112;
    const int bottomSpace = 152;
    const int panelHeight = std::max(250, height - panelTop - bottomSpace);
    const int panelGap = 22;
    const int leftWidth = std::max(360, (width - margin * 2 - panelGap) / 2);
    const int rightX = margin + leftWidth + panelGap;
    const int rightWidth = std::max(360, width - rightX - margin);
    MoveWindow(g_heading, margin, 20, width - margin * 2, 36, TRUE);
    MoveWindow(g_subheading, margin + 1, 60, width - margin * 2, 28, TRUE);
    MoveWindow(g_gameGroup, margin, panelTop, leftWidth, panelHeight, TRUE);
    MoveWindow(g_gameList, margin + 16, panelTop + 30,
        leftWidth - 32, panelHeight - 48, TRUE);
    MoveWindow(g_detailGroup, rightX, panelTop, rightWidth, panelHeight, TRUE);
    MoveWindow(g_activeGame, rightX + 20, panelTop + 36, rightWidth - 40, 30, TRUE);
    MoveWindow(g_status, rightX + 20, panelTop + 76, rightWidth - 40, 78, TRUE);
    MoveWindow(g_progress, rightX + 20, panelTop + 158, rightWidth - 40, 22, TRUE);
    MoveWindow(g_details, rightX + 20, panelTop + 194, rightWidth - 40, panelHeight - 220, TRUE);
    MoveWindow(g_outputText, margin, panelTop + panelHeight + 14, width - margin * 2, 24, TRUE);

    const int buttonY = height - 94;
    const int buttonHeight = 36;
    const int gap = 12;
    const int scanWidth = 128;
    const int launchWidth = 170;
    const int chooseWidth = 202;
    const int openWidth = 188;
    int x = margin;
    MoveWindow(g_scanButton, x, buttonY, scanWidth, buttonHeight, TRUE);
    x += scanWidth + gap;
    MoveWindow(g_launchButton, x, buttonY, launchWidth, buttonHeight, TRUE);
    x += launchWidth + gap;
    MoveWindow(g_chooseButton, x, buttonY, chooseWidth, buttonHeight, TRUE);
    x += chooseWidth + gap;
    MoveWindow(g_openButton, x, buttonY, std::max(openWidth, width - margin - x), buttonHeight, TRUE);
    MoveWindow(g_autoCapture, margin, height - 48, width - margin * 2, 28, TRUE);
}

void create_controls(HWND window) {
    create_child(window, g_heading, L"STATIC", L"TrSS Sample Recorder",
        SS_LEFT | SS_CENTERIMAGE, 0, g_headingFont);
    create_child(window, g_subheading, L"STATIC",
        L"Native Windows capture · no overlay · no game injection · local JSON only",
        SS_LEFT | SS_CENTERIMAGE, 0, g_font);
    create_child(window, g_gameGroup, L"BUTTON", L"Installed game library",
        BS_GROUPBOX, 0, g_font);
    create_child(window, g_gameList, L"LISTBOX", L"",
        WS_VSCROLL | WS_BORDER | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT | LBS_HASSTRINGS,
        IdGameList, g_font);
    create_child(window, g_detailGroup, L"BUTTON", L"Live capture",
        BS_GROUPBOX, 0, g_font);
    create_child(window, g_activeGame, L"STATIC", L"No running game detected",
        SS_LEFT | SS_CENTERIMAGE, 0, g_headingFont);
    create_child(window, g_status, L"STATIC", L"Scanning installed games...",
        SS_LEFT | SS_LEFT, 0, g_font);
    create_child(window, g_progress, PROGRESS_CLASSW, L"",
        PBS_SMOOTH, 0, g_font);
    SendMessageW(g_progress, PBM_SETRANGE32, 0, 100);
    create_child(window, g_details, L"STATIC",
        L"Only visible RGB pixels are recorded. Depth, motion vectors, and private game buffers are not available from generic desktop capture.",
        SS_LEFT | SS_LEFT, 0, g_font);
    create_child(window, g_outputText, L"STATIC", L"",
        SS_LEFT | SS_CENTERIMAGE, 0, g_font);
    create_child(window, g_scanButton, L"BUTTON", L"Scan PC",
        BS_PUSHBUTTON, IdScan, g_font);
    create_child(window, g_launchButton, L"BUTTON", L"Launch selected",
        BS_PUSHBUTTON, IdLaunch, g_font);
    create_child(window, g_chooseButton, L"BUTTON", L"Choose samples folder",
        BS_PUSHBUTTON, IdChooseFolder, g_font);
    create_child(window, g_openButton, L"BUTTON", L"Open samples folder",
        BS_PUSHBUTTON, IdOpenFolder, g_font);
    create_child(window, g_autoCapture, L"BUTTON",
        L"Automatic capture enabled (starts 120 seconds after a game window is detected)",
        BS_AUTOCHECKBOX | WS_TABSTOP, IdAutoCapture, g_font);
    SendMessageW(g_autoCapture, BM_SETCHECK, BST_CHECKED, 0);
    layout_controls(1100, 740);
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        create_controls(window);
        SetTimer(window, kTimerId, kTimerIntervalMs, nullptr);
        do_scan_library();
        set_output_text();
        return 0;
    case WM_SIZE:
        if (g_heading != nullptr) layout_controls(LOWORD(lParam), HIWORD(lParam));
        return 0;
    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        info->ptMinTrackSize.x = 980;
        info->ptMinTrackSize.y = 650;
        return 0;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IdScan:
            do_scan_library();
            return 0;
        case IdLaunch:
            launch_selected_game();
            return 0;
        case IdChooseFolder:
            on_choose_folder();
            return 0;
        case IdOpenFolder:
            on_open_folder();
            return 0;
        case IdGameList:
            if (HIWORD(wParam) == LBN_SELCHANGE) update_game_details();
            return 0;
        case IdAutoCapture:
            if (SendMessageW(g_autoCapture, BM_GETCHECK, 0, 0) != BST_CHECKED) {
                g_writer.close_chunk();
                set_status(L"Automatic capture paused. No new frames will be saved until it is enabled again.");
            } else {
                set_status(L"Automatic capture enabled. The current game remains on its existing warm-up timer.");
            }
            return 0;
        }
        break;
    case WM_TIMER:
        if (wParam == kTimerId) update_monitoring_state();
        return 0;
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLORBTN: {
        HDC dc = reinterpret_cast<HDC>(wParam);
        SetBkColor(dc, RGB(247, 247, 247));
        SetTextColor(dc, RGB(24, 24, 24));
        return reinterpret_cast<LRESULT>(g_backgroundBrush);
    }
    case WM_ERASEBKGND: {
        RECT rect{};
        GetClientRect(window, &rect);
        FillRect(reinterpret_cast<HDC>(wParam), &rect, g_backgroundBrush);
        return 1;
    }
    case WM_CLOSE:
        KillTimer(window, kTimerId);
        g_writer.close();
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wParam, lParam);
}
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand) {
    g_instance = instance;
    SetProcessDPIAware();
    const HRESULT comResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    GdiplusStartupInput startupInput;
    if (GdiplusStartup(&g_gdiplusToken, &startupInput, nullptr) != Ok) {
        MessageBoxW(nullptr, L"Windows could not initialize image encoding.",
            L"TrSS Sample Recorder", MB_OK | MB_ICONERROR);
        if (SUCCEEDED(comResult)) CoUninitialize();
        return 1;
    }
    INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&controls);

    g_outputRoot = fs::path(documents_folder()) / L"TrSS Sample Recorder";
    std::error_code error;
    fs::create_directories(g_outputRoot, error);
    g_font = CreateFontW(-17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    g_headingFont = CreateFontW(-22, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    g_backgroundBrush = CreateSolidBrush(RGB(247, 247, 247));

    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = window_proc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    windowClass.hbrBackground = g_backgroundBrush;
    windowClass.lpszClassName = L"TrSSSampleRecorderWindow";
    if (!RegisterClassExW(&windowClass)) {
        MessageBoxW(nullptr, L"Could not register the TrSS Sample Recorder window.",
            L"TrSS Sample Recorder", MB_OK | MB_ICONERROR);
        GdiplusShutdown(g_gdiplusToken);
        if (SUCCEEDED(comResult)) CoUninitialize();
        return 1;
    }

    g_window = CreateWindowExW(0, windowClass.lpszClassName, L"TrSS Sample Recorder",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT,
        1180, 800, nullptr, nullptr, instance, nullptr);
    if (g_window == nullptr) {
        MessageBoxW(nullptr, L"Could not create the TrSS Sample Recorder window.",
            L"TrSS Sample Recorder", MB_OK | MB_ICONERROR);
        GdiplusShutdown(g_gdiplusToken);
        if (SUCCEEDED(comResult)) CoUninitialize();
        return 1;
    }

    ShowWindow(g_window, showCommand);
    UpdateWindow(g_window);
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }

    if (g_headingFont) DeleteObject(g_headingFont);
    if (g_font) DeleteObject(g_font);
    if (g_backgroundBrush) DeleteObject(g_backgroundBrush);
    GdiplusShutdown(g_gdiplusToken);
    if (SUCCEEDED(comResult)) CoUninitialize();
    return static_cast<int>(message.wParam);
}
