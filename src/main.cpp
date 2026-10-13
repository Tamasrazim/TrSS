#include "frame_image.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <new>
#include <string>
#include <thread>
#include <utility>

#include <windows.h>
#include <commdlg.h>
#include <objbase.h>

namespace {

constexpr wchar_t kWindowClass[] = L"XESS5MainWindow";
constexpr UINT kProcessingFinished = WM_APP + 7;

enum ControlId : int {
    ID_OPEN_IMAGE = 1001,
    ID_BUILD_REFERENCE = 1002,
    ID_SHOW_ORIGINAL = 1003,
    ID_SHOW_REFERENCE = 1004,
    ID_COMPARE = 1005,
};

enum class ViewMode {
    Compare,
    Original,
    Reference,
};

struct ProcessingResult {
    FrameImage image;
    std::wstring error;
};

struct AppState {
    HWND openButton = nullptr;
    HWND buildButton = nullptr;
    HWND originalButton = nullptr;
    HWND referenceButton = nullptr;
    HWND compareButton = nullptr;
    HWND statusLabel = nullptr;

    HBRUSH backgroundBrush = nullptr;
    FrameImage original;
    FrameImage reference;
    std::wstring sourcePath;
    std::wstring status = L"Open a local image to begin.";
    ViewMode view = ViewMode::Compare;
    bool processing = false;
};

[[nodiscard]] AppState* StateFor(HWND window) noexcept {
    return reinterpret_cast<AppState*>(
        GetWindowLongPtrW(window, GWLP_USERDATA)
    );
}

void SetStatus(HWND window, AppState& state, std::wstring message) {
    (void)window; // Status updates are sent directly to the status control.
    state.status = std::move(message);
    if (state.statusLabel != nullptr) {
        SetWindowTextW(state.statusLabel, state.status.c_str());
    }
}

HWND MakeButton(HWND parent, HINSTANCE instance, const wchar_t* label,
                int controlId) {
    return CreateWindowExW(
        0,
        L"BUTTON",
        label,
        WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON,
        0, 0, 140, 34,
        parent,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(controlId)),
        instance,
        nullptr
    );
}

void LayoutControls(HWND window, AppState& state) {
    RECT client{};
    GetClientRect(window, &client);

    constexpr int padding = 16;
    constexpr int gap = 8;
    constexpr int buttonHeight = 34;
    constexpr int statusHeight = 30;

    int x = padding;
    const int y = 14;
    const int clientWidth = static_cast<int>(client.right);
    const int clientHeight = static_cast<int>(client.bottom);
    const int usableWidth = std::max(0, clientWidth - padding * 2);

    auto place = [&](HWND control, int preferredWidth) {
        if (control == nullptr) {
            return;
        }
        const int width = std::min(preferredWidth,
            std::max(0, clientWidth - x - padding));
        MoveWindow(control, x, y, width, buttonHeight, TRUE);
        x += width + gap;
    };

    // Short labels keep the toolbar usable on smaller desktop windows.
    place(state.openButton, 120);
    place(state.buildButton, 158);
    place(state.originalButton, 105);
    place(state.referenceButton, 115);
    place(state.compareButton, 100);

    if (state.statusLabel != nullptr) {
        MoveWindow(
            state.statusLabel,
            padding,
            std::max(0, clientHeight - statusHeight),
            usableWidth,
            statusHeight,
            TRUE
        );
    }
}

std::wstring LeafName(const std::wstring& path) {
    const auto separator = path.find_last_of(L"\\/");
    if (separator == std::wstring::npos) {
        return path;
    }
    return path.substr(separator + 1);
}

void ChooseImage(HWND window, AppState& state) {
    wchar_t path[32768]{};

    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = window;
    dialog.lpstrFile = path;
    dialog.nMaxFile = static_cast<DWORD>(std::size(path));
    dialog.lpstrFilter =
        L"Images\0*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.gif\0"
        L"All files\0*.*\0";
    dialog.nFilterIndex = 1;
    dialog.lpstrTitle = L"Open an image for XESS5 frame inspection";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST |
                   OFN_NOCHANGEDIR | OFN_EXPLORER;

    if (!GetOpenFileNameW(&dialog)) {
        // Closing the picker is normal, not an application error.
        return;
    }

    FrameImage loaded;
    std::wstring error;
    if (!LoadFrameImage(path, loaded, error)) {
        SetStatus(window, state, L"Could not open image: " + error);
        return;
    }

    state.original = std::move(loaded);
    state.reference.clear();
    state.sourcePath = path;
    state.view = ViewMode::Original;

    SetStatus(
        window,
        state,
        L"Loaded " + LeafName(state.sourcePath) + L"  |  " +
        std::to_wstring(state.original.width) + L" x " +
        std::to_wstring(state.original.height) +
        L"  |  Build the reference pass to enable A/B comparison."
    );

    InvalidateRect(window, nullptr, FALSE);
}

void BuildReference(HWND window, AppState& state) {
    if (!state.original.valid()) {
        SetStatus(window, state, L"Open an image before building a comparison.");
        return;
    }
    if (state.processing) {
        SetStatus(window, state, L"The reference pass is already running.");
        return;
    }

    // Allocate the message payload on the UI thread so the worker can always
    // return either an image or an error without owning UI state.
    auto* result = new (std::nothrow) ProcessingResult();
    if (result == nullptr) {
        SetStatus(window, state, L"Not enough memory to start processing.");
        return;
    }

    try {
        FrameImage inputCopy = state.original;
        state.processing = true;
        EnableWindow(state.buildButton, FALSE);
        EnableWindow(state.openButton, FALSE); // Keep the input paired with its result.
        SetStatus(window, state, L"Building a conventional detail reference...");

        std::thread worker(
            [window, source = std::move(inputCopy), result]() mutable {
                try {
                    result->image = MakeDetailReference(source, 1.25f);
                    if (!result->image.valid()) {
                        result->error = L"The reference pass returned no image.";
                    }
                } catch (const std::bad_alloc&) {
                    result->error = L"Not enough memory to process this image.";
                } catch (...) {
                    result->error = L"The reference pass stopped unexpectedly.";
                }

                // The payload is transferred to the UI thread. If the window
                // has already closed, discard it here instead of leaking it.
                if (!PostMessageW(
                        window,
                        kProcessingFinished,
                        0,
                        reinterpret_cast<LPARAM>(result))) {
                    delete result;
                }
            }
        );
        worker.detach();
    } catch (...) {
        state.processing = false;
        EnableWindow(state.buildButton, TRUE);
        EnableWindow(state.openButton, TRUE);
        delete result;
        SetStatus(window, state, L"Could not start the processing worker.");
    }
}

void DrawImageFit(HDC deviceContext, const FrameImage& image, RECT area) {
    if (!image.valid() || area.right <= area.left || area.bottom <= area.top) {
        return;
    }

    const double sourceWidth = static_cast<double>(image.width);
    const double sourceHeight = static_cast<double>(image.height);
    const double areaWidth = static_cast<double>(area.right - area.left);
    const double areaHeight = static_cast<double>(area.bottom - area.top);
    const double scale = std::min(areaWidth / sourceWidth,
                                  areaHeight / sourceHeight);

    const int drawWidth = std::max(1, static_cast<int>(sourceWidth * scale));
    const int drawHeight = std::max(1, static_cast<int>(sourceHeight * scale));
    const int drawX = area.left + ((area.right - area.left) - drawWidth) / 2;
    const int drawY = area.top + ((area.bottom - area.top) - drawHeight) / 2;

    BITMAPINFO bitmapInfo{};
    bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.bmiHeader.biWidth = static_cast<LONG>(image.width);
    // Negative height tells GDI the buffer is top-down, as WIC produced it.
    bitmapInfo.bmiHeader.biHeight = -static_cast<LONG>(image.height);
    bitmapInfo.bmiHeader.biPlanes = 1;
    bitmapInfo.bmiHeader.biBitCount = 32;
    bitmapInfo.bmiHeader.biCompression = BI_RGB;

    SetStretchBltMode(deviceContext, HALFTONE);
    SetBrushOrgEx(deviceContext, 0, 0, nullptr);
    StretchDIBits(
        deviceContext,
        drawX, drawY, drawWidth, drawHeight,
        0, 0, static_cast<int>(image.width), static_cast<int>(image.height),
        image.bgra.data(),
        &bitmapInfo,
        DIB_RGB_COLORS,
        SRCCOPY
    );
}

void DrawPane(HDC deviceContext, RECT pane, const FrameImage& image,
              const wchar_t* label, HFONT labelFont) {
    constexpr int labelHeight = 38;
    SetTextColor(deviceContext, RGB(236, 236, 236));
    SetBkMode(deviceContext, TRANSPARENT);

    RECT labelRect = pane;
    labelRect.bottom = std::min(pane.bottom, pane.top + labelHeight);
    const auto previousFont = SelectObject(deviceContext, labelFont);
    DrawTextW(deviceContext, label, -1, &labelRect,
              DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(deviceContext, previousFont);

    RECT imageRect = pane;
    imageRect.top = std::min(pane.bottom, pane.top + labelHeight);
    InflateRect(&imageRect, -4, -4);
    DrawImageFit(deviceContext, image, imageRect);
}

void PaintCanvas(HWND window, HDC deviceContext, AppState& state) {
    RECT client{};
    GetClientRect(window, &client);

    constexpr int toolbarBottom = 60;
    constexpr int statusHeight = 30;
    RECT canvas{
        0,
        toolbarBottom,
        static_cast<int>(client.right),
        std::max(toolbarBottom, static_cast<int>(client.bottom) - statusHeight)
    };

    HBRUSH canvasBrush = CreateSolidBrush(RGB(14, 14, 14));
    FillRect(deviceContext, &canvas, canvasBrush);
    DeleteObject(canvasBrush);

    if (!state.original.valid()) {
        SetTextColor(deviceContext, RGB(225, 225, 225));
        SetBkMode(deviceContext, TRANSPARENT);

        HFONT headline = CreateFontW(
            -25, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI"
        );
        HFONT regular = reinterpret_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));

        RECT message = canvas;
        message.left += 36;
        message.right -= 36;
        message.top += 24;
        message.bottom -= 24;

        const auto previousFont = SelectObject(deviceContext, headline);
        DrawTextW(deviceContext, L"XESS5  /  FRAME LAB", -1, &message,
                  DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        SelectObject(deviceContext, regular);

        RECT details = message;
        details.top += 42;
        DrawTextW(
            deviceContext,
            L"Open a local image to inspect it. Build the reference pass to compare both versions.\n"
            L"This first pass is a conventional detail filter, not neural rendering.",
            -1,
            &details,
            DT_CENTER | DT_TOP | DT_WORDBREAK
        );
        SelectObject(deviceContext, previousFont);
        DeleteObject(headline);
        return;
    }

    const int width = canvas.right - canvas.left;
    const int height = canvas.bottom - canvas.top;
    if (width <= 0 || height <= 0) {
        return;
    }

    constexpr int outerMargin = 18;
    RECT content{
        canvas.left + outerMargin,
        canvas.top + 10,
        canvas.right - outerMargin,
        canvas.bottom - 10
    };

    HFONT labelFont = CreateFontW(
        -16, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI"
    );

    if (state.view == ViewMode::Original) {
        DrawPane(deviceContext, content, state.original,
                 L"ORIGINAL FRAME", labelFont);
    } else if (state.view == ViewMode::Reference) {
        if (state.reference.valid()) {
            DrawPane(deviceContext, content, state.reference,
                     L"REFERENCE DETAIL  /  NON-NEURAL", labelFont);
        } else {
            SetTextColor(deviceContext, RGB(220, 220, 220));
            RECT textRect = content;
            DrawTextW(deviceContext, L"Build the reference pass first.", -1,
                      &textRect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }
    } else if (state.reference.valid()) {
        const int middle = content.left + (content.right - content.left) / 2;
        RECT leftPane = content;
        leftPane.right = middle - 6;
        RECT rightPane = content;
        rightPane.left = middle + 6;

        DrawPane(deviceContext, leftPane, state.original,
                 L"ORIGINAL", labelFont);
        DrawPane(deviceContext, rightPane, state.reference,
                 L"REFERENCE DETAIL  /  NON-NEURAL", labelFont);

        HPEN separator = CreatePen(PS_SOLID, 1, RGB(80, 80, 80));
        const auto previousPen = SelectObject(deviceContext, separator);
        MoveToEx(deviceContext, middle, content.top, nullptr);
        LineTo(deviceContext, middle, content.bottom);
        SelectObject(deviceContext, previousPen);
        DeleteObject(separator);
    } else {
        DrawPane(deviceContext, content, state.original,
                 L"ORIGINAL  /  BUILD THE REFERENCE TO COMPARE", labelFont);
    }

    DeleteObject(labelFont);
}

LRESULT CALLBACK WindowProcedure(HWND window, UINT message,
                                 WPARAM wParam, LPARAM lParam) {
    AppState* state = StateFor(window);

    if (message == WM_NCCREATE) {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        state = static_cast<AppState*>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(state));
    }

    if (state == nullptr) {
        return DefWindowProcW(window, message, wParam, lParam);
    }

    switch (message) {
    case WM_CREATE: {
        HINSTANCE instance = GetModuleHandleW(nullptr);
        state->backgroundBrush = CreateSolidBrush(RGB(22, 22, 22));

        state->openButton = MakeButton(window, instance, L"Open image", ID_OPEN_IMAGE);
        state->buildButton = MakeButton(window, instance, L"Build reference", ID_BUILD_REFERENCE);
        state->originalButton = MakeButton(window, instance, L"Original", ID_SHOW_ORIGINAL);
        state->referenceButton = MakeButton(window, instance, L"Reference", ID_SHOW_REFERENCE);
        state->compareButton = MakeButton(window, instance, L"Compare A/B", ID_COMPARE);
        state->statusLabel = CreateWindowExW(
            0, L"STATIC", state->status.c_str(),
            WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
            0, 0, 100, 30, window, nullptr, instance, nullptr
        );

        const HWND controls[] = {
            state->openButton, state->buildButton, state->originalButton,
            state->referenceButton, state->compareButton, state->statusLabel
        };
        for (HWND control : controls) {
            if (control == nullptr) {
                return -1;
            }
            SendMessageW(control, WM_SETFONT,
                         reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)),
                         TRUE);
        }

        LayoutControls(window, *state);
        return 0;
    }

    case WM_SIZE:
        LayoutControls(window, *state);
        InvalidateRect(window, nullptr, FALSE);
        return 0;

    case WM_GETMINMAXINFO: {
        auto* limits = reinterpret_cast<MINMAXINFO*>(lParam);
        limits->ptMinTrackSize.x = 780;
        limits->ptMinTrackSize.y = 520;
        return 0;
    }

    case WM_COMMAND:
        if (HIWORD(wParam) == BN_CLICKED) {
            switch (LOWORD(wParam)) {
            case ID_OPEN_IMAGE:
                ChooseImage(window, *state);
                return 0;
            case ID_BUILD_REFERENCE:
                BuildReference(window, *state);
                return 0;
            case ID_SHOW_ORIGINAL:
                state->view = ViewMode::Original;
                InvalidateRect(window, nullptr, FALSE);
                return 0;
            case ID_SHOW_REFERENCE:
                if (!state->reference.valid()) {
                    SetStatus(window, *state, L"Build the reference pass before viewing it.");
                } else {
                    state->view = ViewMode::Reference;
                    InvalidateRect(window, nullptr, FALSE);
                }
                return 0;
            case ID_COMPARE:
                if (!state->reference.valid()) {
                    SetStatus(window, *state, L"Build the reference pass before comparing A/B.");
                } else {
                    state->view = ViewMode::Compare;
                    InvalidateRect(window, nullptr, FALSE);
                }
                return 0;
            default:
                break;
            }
        }
        break;

    case kProcessingFinished: {
        auto* result = reinterpret_cast<ProcessingResult*>(lParam);
        if (result == nullptr) {
            return 0;
        }

        state->processing = false;
        EnableWindow(state->buildButton, TRUE);
        EnableWindow(state->openButton, TRUE);

        if (!result->error.empty()) {
            SetStatus(window, *state, L"Reference pass failed: " + result->error);
        } else {
            state->reference = std::move(result->image);
            state->view = ViewMode::Compare;
            SetStatus(
                window,
                *state,
                L"Reference ready. This is a conventional sharpen baseline, not an AI model. "
                L"No game files have been touched."
            );
        }

        delete result;
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }

    case WM_CTLCOLORSTATIC: {
        HDC deviceContext = reinterpret_cast<HDC>(wParam);
        SetTextColor(deviceContext, RGB(205, 205, 205));
        SetBkMode(deviceContext, TRANSPARENT);
        return reinterpret_cast<LRESULT>(state->backgroundBrush);
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT: {
        PAINTSTRUCT paint{};
        HDC deviceContext = BeginPaint(window, &paint);
        PaintCanvas(window, deviceContext, *state);
        EndPaint(window, &paint);
        return 0;
    }

    case WM_DESTROY:
        if (state->backgroundBrush != nullptr) {
            DeleteObject(state->backgroundBrush);
            state->backgroundBrush = nullptr;
        }
        PostQuitMessage(0);
        return 0;

    default:
        break;
    }

    return DefWindowProcW(window, message, wParam, lParam);
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand) {
    const HRESULT comResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool shouldUninitializeCom = SUCCEEDED(comResult);

    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = reinterpret_cast<HBRUSH>(
        GetStockObject(BLACK_BRUSH)
    );
    windowClass.lpszClassName = kWindowClass;

    if (RegisterClassExW(&windowClass) == 0) {
        if (shouldUninitializeCom) {
            CoUninitialize();
        }
        MessageBoxW(nullptr, L"Could not register the XESS5 window class.",
                    L"XESS5", MB_OK | MB_ICONERROR);
        return 1;
    }

    AppState state;
    HWND window = CreateWindowExW(
        0,
        kWindowClass,
        L"XESS5  |  Frame Lab",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        1280, 800,
        nullptr,
        nullptr,
        instance,
        &state
    );

    if (window == nullptr) {
        if (shouldUninitializeCom) {
            CoUninitialize();
        }
        MessageBoxW(nullptr, L"Could not create the XESS5 application window.",
                    L"XESS5", MB_OK | MB_ICONERROR);
        return 1;
    }

    ShowWindow(window, showCommand);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }

    if (shouldUninitializeCom) {
        CoUninitialize();
    }
    return static_cast<int>(message.wParam);
}
