import io
import cv2
import base64
import numpy as np
from PIL import Image
from typing import Any, Optional


def resize_image(
    img: Optional[Image.Image],
    new_longest: int,
) -> Image.Image:
    """Resize image based on its longest side (either width or height).

    Args:
        img (Image.Image): the input image.
        new_longest (int): the pixel length of image of resize.

    Returns:
        Image.Image: the image after resize.
    """
    if not img:
        return None

    try:
        ratio = max(img.width, img.height) / new_longest
        new_size = (int(img.width / ratio), int(img.height / ratio))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    except Exception:
        img_resized = img

    return img_resized


def trim_empty_space(
    pil_img: Image.Image,
    expansion_buffer: int = 10,
) -> Optional[Image.Image]:
    """Trim the empty spaces if the image has empty spaces surrounding the actual image.

    Args:
        pil_img (Image.Image): the given image.
        expansion_buffer (int, optional): the padding margin extending detected edges.
            Defaults to 10 pixel.

    Returns:
        Image.Image: the trimed image.
    """
    cv_img = np.array(pil_img)[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - expansion_buffer)
    y = max(0, y - expansion_buffer)
    w = min(cv_img.shape[1] - x, w + 2 * expansion_buffer)
    h = min(cv_img.shape[0] - y, h + 2 * expansion_buffer)
    cropped_image = cv_img[y : y + h, x : x + w]
    return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


def pil_to_base64(
    pil_img: Image.Image,
    fmt: str = "jpeg",
    logger: Any = None,
) -> str:
    """Encode PIL image into base64.

    Args:
        pil_img (Image.Image): Input image.
        fmt (str, optional): format of the input image. Defaults to 'png'.

    Returns:
        str: the base64-encoded string for the given image.
    """
    try:
        image = pil_img
        width, height = image.size

        if width < 10 or height < 10:
            image = image.resize((max(width, 10), max(height, 10)))
            width, height = image.size

        ratio = max(width, height) / min(width, height)
        if ratio > 200:
            if width > height:
                new_height = max(10, int(width / 199))
                image = image.resize((width, new_height))
            else:
                new_width = max(10, int(height / 199))
                image = image.resize((new_width, height))
            width, height = image.size

        max_pixels = 7680 * 4320
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        if fmt.lower() in {"jpg", "jpeg"}:
            image = image.convert("RGB")
            quality = 90
            while quality > 10:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=quality)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                if len(img_str) < 10 * 1024 * 1024:
                    return img_str
                quality -= 10
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        buffered = io.BytesIO()
        image.save(buffered, format=fmt)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error with image encoding: {e}")
        return ""


def get_pil_from_pdf_page(
    page,
    resize_width: int = 1344,
    dpi: int = 360,
) -> Optional[Image.Image]:
    """Get PIL image from pdf pages after all preprocessings

    Args:
        page (_type_): PyMuPDF page object
        resize_width (int, optional): normalization size width. Defaults to 1344.
        dpi (int): Dots Per Inch, reflecting the resolution of printed images.
            Defaults to 360.

    Returns:
        Image.Image: the converted image from pdf page
    """
    pixmap = page.get_pixmap(alpha=False, dpi=dpi)
    image_data = pixmap.tobytes()
    image_stream = io.BytesIO(image_data)
    pil_img = Image.open(image_stream).convert("RGB")
    trimmed_img = trim_empty_space(pil_img)
    return resize_image(trimmed_img, resize_width)
    
