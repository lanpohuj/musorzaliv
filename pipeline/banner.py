"""
Banner generation module for CS2 Clips.
Generates or loads promo banners for overlays.
"""

import os
from PIL import Image, ImageDraw, ImageFont


def get_font(font_name: str, size: int):
    """Attempt to load a system or local font, fallback to default."""
    font_candidates = [
        font_name,
        "/nix/store/akycp4lqk3hkyg3chwf7jwc14x7vd3d8-kitty-0.49.1/lib/kitty/fonts/DejaVuSans-Bold.ttf",
        "/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans-Bold.ttf",
        "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def create_or_get_banner(config: dict, output_path: str = "temp/banner.png") -> str:
    """
    Returns path to a ready-to-use RGBA PNG banner.
    If custom_image is specified and exists, uses that.
    Otherwise renders a modern CS2 gambling/roulette promo card.
    """
    banner_cfg = config.get("banner", {})
    custom_img = banner_cfg.get("custom_image") or "assets/banner.png"
    
    if custom_img and os.path.exists(custom_img):
        return custom_img

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    width = banner_cfg.get("width", 960)
    height = banner_cfg.get("height", 140)
    title = banner_cfg.get("title", "🔥 ПРОМОКОД: CS2WIN (+2$ FREE)")
    subtitle = banner_cfg.get("subtitle", "KEYDROP / CSFAIL • ХАЛЯВНИЙ БАЛАНС")

    # High-resolution canvas for super crisp text & anti-aliasing
    scale = 2
    canvas_w = width * scale
    canvas_h = height * scale
    
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer glow / border
    border_radius = 24 * scale
    draw.rounded_rectangle(
        [(4 * scale, 4 * scale), (canvas_w - 4 * scale, canvas_h - 4 * scale)],
        radius=border_radius,
        fill=(15, 17, 24, 235),
        outline=(255, 150, 0, 240),
        width=4 * scale
    )

    # Accent left strip / tag
    tag_w = 160 * scale
    tag_h = 42 * scale
    tag_x = 28 * scale
    tag_y = 22 * scale
    draw.rounded_rectangle(
        [(tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h)],
        radius=8 * scale,
        fill=(255, 120, 0, 255)
    )

    tag_font = get_font("DejaVu Sans", int(18 * scale))
    draw.text(
        (tag_x + 14 * scale, tag_y + 8 * scale),
        "CS2 BONUS",
        font=tag_font,
        fill=(255, 255, 255, 255)
    )

    # Title text
    title_font = get_font("DejaVu Sans", int(28 * scale))
    draw.text(
        (tag_x + tag_w + 24 * scale, tag_y + 4 * scale),
        title,
        font=title_font,
        fill=(255, 225, 60, 255)
    )

    # Subtitle text
    sub_font = get_font("DejaVu Sans", int(20 * scale))
    draw.text(
        (tag_x + 6 * scale, tag_y + tag_h + 18 * scale),
        subtitle,
        font=sub_font,
        fill=(220, 225, 235, 255)
    )

    # Downscale for nice smooth anti-aliased edges
    final_img = img.resize((width, height), Image.Resampling.LANCZOS)
    final_img.save(output_path, "PNG")
    return output_path
