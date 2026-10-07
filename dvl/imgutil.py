from PIL import Image, ImageOps


def to_rgb_resized(img: Image.Image, max_side: int = 512) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def ahash(img: Image.Image) -> int:
    small = img.convert("L").resize((8, 8), Image.BILINEAR)
    px = list(small.tobytes())
    avg = sum(px) / 64
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
