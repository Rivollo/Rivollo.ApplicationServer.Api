"""Pure colour-math helpers.

The single most common bug when recolouring a GLB is forgetting that
``baseColorFactor`` is stored in **linear** space while a ``#RRGGBB`` hex from a
colour picker is **sRGB**. Multiplying a linear factor by an sRGB value gives a
washed-out result. Everything here keeps those two spaces explicit.
"""
from __future__ import annotations

Rgb = tuple[int, int, int]


def hex_to_rgb(value: str) -> Rgb:
    """'#C0182B' or 'C0182B' -> (192, 24, 43)."""
    v = value.strip().lstrip("#")
    if len(v) == 3:  # short form #abc
        v = "".join(ch * 2 for ch in v)
    if len(v) != 6:
        raise ValueError(f"Invalid hex colour: {value!r}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def rgb_to_hex(rgb: Rgb) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _srgb_channel_to_linear(c: float) -> float:
    """Accurate IEC 61966-2-1 sRGB -> linear transfer (c in 0..1)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_channel_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def hex_to_linear_factor(value: str, alpha: float = 1.0) -> list[float]:
    """Hex sRGB -> glTF linear ``baseColorFactor`` [r, g, b, a]."""
    r, g, b = hex_to_rgb(value)
    return [
        _srgb_channel_to_linear(r / 255),
        _srgb_channel_to_linear(g / 255),
        _srgb_channel_to_linear(b / 255),
        alpha,
    ]


def linear_factor_to_hex(factor: list[float] | None) -> str:
    """glTF linear ``baseColorFactor`` -> displayable sRGB hex."""
    if not factor:
        return "#FFFFFF"
    r, g, b = factor[0], factor[1], factor[2]
    return rgb_to_hex(
        (
            _linear_channel_to_srgb(r) * 255,
            _linear_channel_to_srgb(g) * 255,
            _linear_channel_to_srgb(b) * 255,
        )
    )


# --------------------------------------------------------------------------- #
# HSL + brightness adjustment (used by the per-part brightness slider)
# --------------------------------------------------------------------------- #
def _rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """r,g,b in 0..1 -> h,s,l in 0..1."""
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h / 6, s, l


def _hue_to_rgb(p: float, q: float, t: float) -> float:
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    if s == 0:
        return l, l, l
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (
        _hue_to_rgb(p, q, h + 1 / 3),
        _hue_to_rgb(p, q, h),
        _hue_to_rgb(p, q, h - 1 / 3),
    )


def adjust_brightness_hex(value: str, brightness: float) -> str:
    """Shift a colour's HSL lightness by a slider factor.

    ``brightness`` == 1.0 leaves the colour unchanged. Above 1 moves the
    lightness toward white; below 1 moves it toward black. Hue and saturation
    are preserved, so only perceived brightness changes.
    """
    if abs(brightness - 1.0) < 1e-6:
        return value
    r, g, b = (c / 255 for c in hex_to_rgb(value))
    h, s, l = _rgb_to_hsl(r, g, b)
    if brightness >= 1:
        l2 = l + (1 - l) * (brightness - 1)
    else:
        l2 = l * brightness
    l2 = max(0.0, min(1.0, l2))
    nr, ng, nb = _hsl_to_rgb(h, s, l2)
    return rgb_to_hex((nr * 255, ng * 255, nb * 255))
