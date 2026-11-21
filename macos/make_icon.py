"""Draw the original Twitch Shorts macOS icon and prepare an iconset."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFilter

size = 1024
icon = Image.new("RGBA", (size, size))
shadow = Image.new("RGBA", icon.size)
ImageDraw.Draw(shadow).rounded_rectangle((112, 134, 912, 934), radius=185, fill=(10, 2, 25, 100))
icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(27)))

mask = Image.new("L", icon.size)
ImageDraw.Draw(mask).rounded_rectangle((112, 105, 912, 905), radius=185, fill=255)
plate = Image.new("RGBA", icon.size)
paint = ImageDraw.Draw(plate)
for y in range(size):
    t = min(1, max(0, (y - 105) / 800))
    color = tuple(round(a * (1 - t) + b * t) for a, b in zip((124, 64, 227), (39, 21, 91)))
    paint.line((0, y, size, y), fill=color)
plate.putalpha(mask)
icon.alpha_composite(plate)

draw = ImageDraw.Draw(icon)
draw.rounded_rectangle((112, 105, 912, 905), radius=185, outline="#9964ed", width=4)
draw.rounded_rectangle((321, 244, 703, 783), radius=76, fill="#281543")
draw.rounded_rectangle((321, 232, 703, 767), radius=76, fill="#f6efff")
draw.rounded_rectangle((345, 256, 679, 743), radius=56, fill="#412073")
draw.rounded_rectangle((345, 256, 679, 498), radius=56, fill="#6839aa")
draw.rectangle((345, 402, 679, 497), fill="#6839aa")
draw.line((345, 498, 679, 498), fill="#bc91ff", width=10)
draw.polygon([(466, 554), (580, 619), (466, 684)], fill="#fff166")
draw.ellipse((472, 297, 552, 377), fill="#f6efff")
draw.rounded_rectangle((434, 395, 590, 470), radius=36, fill="#f6efff")

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
for points in (16, 32, 128, 256, 512):
    for scale, suffix in ((1, ""), (2, "@2x")):
        pixels = points * scale
        icon.resize((pixels, pixels), Image.Resampling.LANCZOS).save(
            folder / f"icon_{points}x{points}{suffix}.png"
        )
