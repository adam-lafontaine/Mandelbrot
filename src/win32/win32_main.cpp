#include "win32_main.hpp"
#include "../app/app.hpp"
#include "../utils/stopwatch.hpp"

#include <string>
#include <thread>

#define CHECK_LEAKS
#if defined(_WIN32) && defined(_DEBUG) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

constexpr auto MAIN_WINDOW = "MainWindow";

constexpr auto WINDOW_TITLE_LEN = std::char_traits<char>::length(app::APP_TITLE) + 1;
constexpr auto MAIN_WINDOW_NAME_LEN = std::char_traits<char>::length(MAIN_WINDOW) + WINDOW_TITLE_LEN;

GlobalVariable wchar_t WINDOW_TITLE[WINDOW_TITLE_LEN];
GlobalVariable wchar_t MAIN_WINDOW_NAME[MAIN_WINDOW_NAME_LEN];


void load_window_title()
{
    swprintf_s(WINDOW_TITLE, L"%hs", app::APP_TITLE);
}


void load_main_window_name()
{
    swprintf_s(MAIN_WINDOW_NAME, L"%hs%hs", app::APP_TITLE, MAIN_WINDOW);
}


// size of window
// bitmap buffer will be scaled to these dimensions Windows (StretchDIBits)
constexpr int WINDOW_AREA_WIDTH = app::BUFFER_WIDTH;
constexpr int WINDOW_AREA_HEIGHT = app::BUFFER_HEIGHT;

// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 60.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

// flag to signal when the application should terminate
GlobalVariable b32 g_running = false;

// contains the memory that the application will draw to
GlobalVariable win32::BitmapBuffer g_back_buffer = {};
GlobalVariable WINDOWPLACEMENT g_window_placement = { sizeof(g_window_placement) };


u32 platform_to_color_32(u8 red, u8 green, u8 blue)
{
    return red << 16 | green << 8 | blue;
}


void platform_signal_stop()
{
    g_running = false;
}


static void end_program(app::AppMemory& memory)
{
    g_running = false;
    app::end_program(memory);
}



namespace win32
{
    static void resize_offscreen_buffer(BitmapBuffer& buffer, u32 width, u32 height)
    {
        if (buffer.memory)
        {
            VirtualFree(buffer.memory, 0, MEM_RELEASE);
        }

        int iwidth = (int)(width);
        int iheight = (int)(height);

        buffer.width = iwidth;
        buffer.height = iheight;

        buffer.info.bmiHeader.biSize = sizeof(buffer.info.bmiHeader);
        buffer.info.bmiHeader.biWidth = iwidth;
        buffer.info.bmiHeader.biHeight = -iheight; // top down
        buffer.info.bmiHeader.biPlanes = 1;
        buffer.info.bmiHeader.biBitCount = buffer.bytes_per_pixel * 8; // 3 bytes + 1 byte padding
        buffer.info.bmiHeader.biCompression = BI_RGB;

        int bitmap_memory_size = iwidth * iheight * buffer.bytes_per_pixel;

        buffer.memory = (u8*)VirtualAlloc(0, bitmap_memory_size, MEM_COMMIT, PAGE_READWRITE);

    }


    static void display_buffer_in_window(BitmapBuffer& buffer, HDC device_context)
    {
        StretchDIBits(
            device_context,
            0, 0, WINDOW_AREA_WIDTH, WINDOW_AREA_HEIGHT, // dst
            0, 0, buffer.width, buffer.height, // src
            buffer.memory,
            &(buffer.info),
            DIB_RGB_COLORS, SRCCOPY
        );
    }


    static void toggle_fullscreen(HWND window)
    {
        // https://devblogs.microsoft.com/oldnewthing/20100412-00/?p=14353
        DWORD dwStyle = GetWindowLong(window, GWL_STYLE);
        if (dwStyle & WS_OVERLAPPEDWINDOW)
        {
            MONITORINFO mi = { sizeof(mi) };
            if (GetWindowPlacement(window, &g_window_placement) && GetMonitorInfo(MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY), &mi))
            {
                SetWindowLong(window, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(window, HWND_TOP,
                    mi.rcMonitor.left, mi.rcMonitor.top,
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            }
        }
        else
        {
            SetWindowLong(window, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(window, &g_window_placement);
            SetWindowPos(window, NULL, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        }
    }


    static WindowDims get_window_dimensions(HWND window)
    {
        RECT client_rect;
        GetClientRect(window, &client_rect);

        return {
            client_rect.right - client_rect.left,
            client_rect.bottom - client_rect.top
        };
    }


    bool handle_alt_key_down(MSG const& msg)
    {
        auto const alt_key_down = [](MSG const& msg) { return (msg.lParam & (1u << 29)); };

        if (!alt_key_down(msg))
        {
            return false;
        }

        switch (msg.wParam)
        {
        case VK_RETURN:
            toggle_fullscreen(msg.hwnd);
            return true;

        case VK_F4:
            g_running = false;
            return true;
        }

        return false;
    }

}



static void allocate_app_memory(app::AppMemory& memory, win32::MemoryState& win32_memory)
{
    memory.permanent_storage_size = Megabytes(256);

    size_t total_size = memory.permanent_storage_size;

    LPVOID base_address = 0;

    memory.permanent_storage = VirtualAlloc(base_address, (SIZE_T)total_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

    win32_memory.total_size = total_size;
    win32_memory.memory_block = memory.permanent_storage;
}


static app::ScreenBuffer make_app_pixel_buffer()
{
    app::ScreenBuffer buffer = {};

    buffer.memory = g_back_buffer.memory;
    buffer.width = g_back_buffer.width;
    buffer.height = g_back_buffer.height;
    buffer.bytes_per_pixel = g_back_buffer.bytes_per_pixel;

    return buffer;
}


#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance


// Forward declarations of functions included in this code module:
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);


static WNDCLASSEXW make_window_class(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_APPLICATIONWIN32);
    wcex.lpszClassName = MAIN_WINDOW_NAME;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_ICON_SMALL));

    return wcex;
}


static HWND make_window(HINSTANCE hInstance)
{
    int extra_width = 16;
    int extra_height = 59;

    return CreateWindowW( // https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-createwindowexa
        MAIN_WINDOW_NAME,
        WINDOW_TITLE,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        WINDOW_AREA_WIDTH + extra_width,
        WINDOW_AREA_HEIGHT + extra_height,
        nullptr,
        nullptr,
        hInstance,
        nullptr);
}






int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{


#if defined(_WIN32) && defined(_DEBUG) && defined(CHECK_LEAKS)
    int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
    dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
    dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
    _CrtSetDbgFlag(dbgFlags);
#endif

    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    load_main_window_name();
    load_window_title();

    WNDCLASSEXW window_class = make_window_class(hInstance);
    if (!RegisterClassExW(&window_class))
    {
        return 0;
    }

    // Perform application initialization:
    hInst = hInstance; // Store instance handle in our global variable

    HWND window = make_window(hInstance);

    if (!window)
    {
        return 0;
    }

    ShowWindow(window, nCmdShow);
    UpdateWindow(window);

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_APPLICATIONWIN32));
    HDC device_context = GetDC(window);
    auto window_dims = win32::get_window_dimensions(window);

    win32::MemoryState win32_memory = {};
    app::AppMemory app_memory = {};

    
    allocate_app_memory(app_memory, win32_memory);
    if (!app_memory.permanent_storage)
    {
        return 0;
    }    

    win32::resize_offscreen_buffer(g_back_buffer, app::BUFFER_WIDTH, app::BUFFER_HEIGHT);
    if (!g_back_buffer.memory)
    {
        return 0;
    }

    SetWindowTextW(window, WINDOW_TITLE);

    auto app_pixel_buffer = make_app_pixel_buffer();    

    g_running = app::initialize_memory(app_memory, app_pixel_buffer);
    

    Input input[2] = {};

    bool in_current = 0;
    bool in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;
    wchar_t title_buffer[30];
    r64 ms_elapsed = 0.0;
    r64 title_refresh_ms = 500.0;

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();

        if (ms_elapsed >= title_refresh_ms)
        {
            ms_elapsed = 0.0;
            swprintf_s(title_buffer, L"%s %d", WINDOW_TITLE, (int)frame_ms_elapsed);
            SetWindowTextW(window, title_buffer);
        }

        auto sleep_ms = (u32)(TARGET_MS_PER_FRAME - frame_ms_elapsed);
        if (frame_ms_elapsed < TARGET_MS_PER_FRAME && sleep_ms > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            while (frame_ms_elapsed < TARGET_MS_PER_FRAME)
            {
                frame_ms_elapsed = sw.get_time_milli();
            }
        }

        ms_elapsed += frame_ms_elapsed;

        sw.start();
    };


    sw.start();
    while (g_running)
    {
        // does not miss frames but slows animation
        input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //new_input->dt_frame = frame_ms_elapsed / 1000.0f;

        win32::process_keyboard_input(input[in_old].keyboard, input[in_current].keyboard);
        win32::process_mouse_input(window, input[in_old].mouse, input[in_current].mouse);
        app::update_and_render(app_memory, input[in_current], app_pixel_buffer);

        wait_for_framerate();

        win32::display_buffer_in_window(g_back_buffer, device_context);
        
        // swap inputs
        in_current = in_old;
        in_old = !in_old;
    }


    ReleaseDC(window, device_context);

    return 0;
}


//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;

            case IDM_EXIT: // File > Exit
                DestroyWindow(hWnd);
                g_running = false;
                break;

            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            win32::display_buffer_in_window(g_back_buffer, hdc);
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY: // X button
        PostQuitMessage(0);
        g_running = false;
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}


// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
