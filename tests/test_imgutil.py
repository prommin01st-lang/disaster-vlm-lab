from PIL import Image

from dvl.imgutil import ahash, hamming, to_rgb_resized


def test_resize_keeps_aspect_and_limits_side():
    out = to_rgb_resized(Image.new("RGB", (2000, 1000)))
    assert out.size == (512, 256) and out.mode == "RGB"


def test_small_image_not_upscaled():
    assert to_rgb_resized(Image.new("RGB", (100, 80))).size == (100, 80)


def test_rgba_composited_on_white():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))  # โปร่งใสทั้งภาพ
    out = to_rgb_resized(img)
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)


def test_grayscale_and_palette():
    assert to_rgb_resized(Image.new("L", (10, 10))).mode == "RGB"
    assert to_rgb_resized(Image.new("P", (10, 10))).mode == "RGB"


def test_exif_rotation_applied():
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    img.info["exif"] = exif.tobytes()
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = to_rgb_resized(Image.open(io.BytesIO(buf.getvalue())))
    assert out.size == (20, 40)


def test_ahash_near_duplicate():
    a = Image.linear_gradient("L").convert("RGB")
    b = a.resize((128, 128))
    c = a.rotate(180)
    assert hamming(ahash(a), ahash(b)) <= 4
    assert hamming(ahash(a), ahash(c)) > 20
