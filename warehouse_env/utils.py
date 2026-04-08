"""
Shared Utilities — Warehouse Order Fulfillment
================================================
Cross-platform helpers for UTF-8 terminal output and safe emoji rendering.
"""

from __future__ import annotations

import io
import os
import sys


# ── UTF-8 stdout enforcement ────────────────────────────────────────

def ensure_utf8_stdout() -> None:
    """Force stdout/stderr to UTF-8 encoding on Windows.

    Windows terminals (cmd, PowerShell) default to cp1252 or similar,
    which crashes on box-drawing chars and emoji. This fixes it.
    Call once at the top of any script that prints Unicode.
    """
    if sys.platform == "win32":
        # Try setting the console to UTF-8 via the Windows API
        try:
            os.system("")  # enables VT100 escape sequences on Win10+
        except Exception:
            pass

    # Wrap stdout/stderr if they aren't already UTF-8
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "encoding") and stream.encoding and stream.encoding.lower() != "utf-8":
            try:
                wrapped = io.TextIOWrapper(
                    stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
                )
                setattr(sys, stream_name, wrapped)
            except Exception:
                pass  # If it fails (e.g. no .buffer), leave as-is


# ── Safe unicode / emoji helper ─────────────────────────────────────

def _can_encode_unicode() -> bool:
    """Check whether stdout can handle multi-byte Unicode."""
    try:
        enc = sys.stdout.encoding or "ascii"
        "\u2554\u2550\u2557".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Module-level cache
_UNICODE_OK: bool | None = None


def unicode_ok() -> bool:
    """Return True if the current stdout supports box-drawing / emoji."""
    global _UNICODE_OK
    if _UNICODE_OK is None:
        _UNICODE_OK = _can_encode_unicode()
    return _UNICODE_OK


# ── Emoji map with ASCII fallbacks ──────────────────────────────────

_EMOJI_MAP: dict[str, tuple[str, str]] = {
    # key: (unicode_version, ascii_fallback)
    "check":    ("\u2705", "[OK]"),
    "cross":    ("\u274c", "[X]"),
    "arrow":    ("\u25b6", ">>"),
    "trophy":   ("\U0001f3c6", "[#1]"),
    "folder":   ("\U0001f4c1", "[DIR]"),
    "clock":    ("\u23f3", "[..]"),
    "rocket":   ("\U0001f680", "[GO]"),
    "warn":     ("\u26a0\ufe0f", "[!]"),
    "star":     ("\u2605", "*"),
    "dash":     ("\u2500", "-"),
    "heavy_eq": ("\u2550", "="),
    "tl":       ("\u2554", "+"),
    "tr":       ("\u2557", "+"),
    "bl":       ("\u255a", "+"),
    "br":       ("\u255d", "+"),
    "ml":       ("\u2560", "+"),
    "mr":       ("\u2563", "+"),
    "vl":       ("\u2551", "|"),
    "hl":       ("\u2500", "-"),
    "bar_full": ("\u2588", "#"),
    "bar_empty":("\u2591", "."),
}


def icon(name: str) -> str:
    """Return a Unicode icon if supported, else an ASCII fallback.

    Usage:
        print(f"{icon('check')} All tests passed!")
        # → "✅ All tests passed!"   (on good terminals)
        # → "[OK] All tests passed!" (on cp1252 terminals)
    """
    entry = _EMOJI_MAP.get(name)
    if entry is None:
        return name  # passthrough unknown keys
    return entry[0] if unicode_ok() else entry[1]


def box_char(name: str) -> str:
    """Alias for icon() — used for box-drawing characters."""
    return icon(name)
