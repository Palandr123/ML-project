from typing import Optional
import math

from PIL import ImageOps, Image


def resize_images(images: list[Image.Image], size: Optional[int] = 512) -> list[Image.Image]:
    """
    Resize a list of images to a specified size.
    
    Args:
        images: list[Image.Image] - a list of PIL Image objects to be resized.
        size: Optional[int] - the size (width and height) of each resized image

    Returns:
        Image.Image: A list of resized PIL Images.
    """
    return [ImageOps.fit(image, (size, size), Image.LANCZOS) for image in images]


def create_canvas(num_images: int, size: Optional[int] = 512) -> tuple[Image.Image, tuple[int, int]]:
    """
    Create a canvas for the final image with dimensions based on the number of images.
    
    Args:
        num_images: int - the number of images in canvas.
        size: Optional[int] - the size (width and height) of each resized image

    Returns:
        Image.Image: a canvas
        tuple[int, int]: number of rows and cols in grid
    """
    rows = math.isqrt(num_images)
    cols = math.ceil(num_images / rows)
    canvas_width = size * cols
    canvas_height = size * rows
    return Image.new('RGB', (canvas_width, canvas_height)), (rows, cols)


def concat_images(images: list[Image.Image], size: Optional[int] = 512) -> Image.Image:
    """
    Concatenate a list of images into a single image grid.

    Args:
        images: list[Image.Image] - a list of PIL Image objects to be concatenated
        size: Optional[int] - the size (width and height) of each resized image

    Returns:
        Image.Image: A PIL Image containing the concatenated images.
    """
    resized_images = resize_images(images, size)
    canvas, shape = create_canvas(len(resized_images), size)
                     
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = size * col, size * row
            idx = row * shape[1] + col
            canvas.paste(resized_images[idx], offset)

    return canvas