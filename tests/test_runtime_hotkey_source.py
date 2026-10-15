"""Regression test for the in-game menu shortcut.

Windows normally dispatches Alt+key combinations as WM_SYSKEYDOWN rather than
WM_KEYDOWN. Keep both paths accepted so Ctrl+Alt+Page Up can open the menu.
"""
from pathlib import Path


def test_ctrl_alt_page_up_accepts_system_keydown() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "native"
        / "runtime"
        / "src"
        / "trss_runtime.cpp"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "(message == WM_KEYDOWN || message == WM_SYSKEYDOWN)" in source
    assert "wParam == VK_PRIOR" in source
    assert "(GetKeyState(VK_CONTROL) & 0x8000)" in source
    assert "(GetKeyState(VK_MENU) & 0x8000)" in source
