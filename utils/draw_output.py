import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import math

def chebyshev_distance(x1, x2, y1, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def draw_box(array, x_center, y_center, radius, thickness=5):
    """
    :param array: (HxWxRBGA) numpy array
    :param x_center: center x-coordinate of petri dish center
    :param y_center: center y-coordinate of petri dish center
    :param radius: radius of the petri dish
    :param thickness: desired thickness of box border. Default = 5
    :return: Original array with a red border representing the bounding box.
    """
    length = array.shape[0]
    width = array.shape[1]
    final_array = np.copy(array)
    for x in range(width):
        for y in range(length):
            # take pixels in the "thickness" neighborhood around the outside of the box
            if radius - thickness < chebyshev_distance(x, x_center, y, y_center) < radius + thickness:
                final_array[y, x] = [255, 0, 0, 255]  # red RGB value
    return final_array


def draw_circle(array, x_center, y_center, radius, thickness=1):
    """
    :param array: (HxWxRBGA) numpy array
    :param x_center: center x-coordinate of petri dish
    :param y_center: center y-coordinate of petri dish
    :param radius: radius of the petri dish
    :param thickness: desired thickness of circle border. Default = 1
    :return: Original array with a blue border representing the petri dish.
    """
    length = array.shape[0]
    width = array.shape[1]
    final_array = np.copy(array)
    for x in range(width):
        for y in range(length):
            # take pixels in the "thickness" neighborhood around the outside of the box
            if radius - thickness < math.dist([x, y], [x_center, y_center]) < radius + thickness:
                final_array[y, x] = [0, 0, 255, 255]  # blue RGB value
    return final_array

def draw_prediction(original_img, mask, x_center, y_center, radius, jac, result_folder, filename):
    """
    :param original_img: original slime mold image passed as a PIL Image
    :param mask: mask with same size as original image passed as a PIL Image
    :param x_center: x-coordinate of petri dish center
    :param y_center: y-coordinate of petri dish center
    :param radius: radius of petri dish
    :param jac: predicted jaccard from regression sigmoid function
    :param result_folder: folder to store final output
    saves original image with slime mold coloring, bounding box, and predicted jaccard.
    """
    # load
    background = original_img.convert("RGBA")
    mask = mask.convert("RGBA")
    # create pink_mask
    position_blob = np.array(mask.convert("L")) > 128
    blob4 = np.repeat(position_blob[:, :, np.newaxis], 4, axis=2)
    pink_image = PIL.Image.new("RGBA", background.size, color=(255, 150, 150, 180))
    mask_new = PIL.Image.fromarray(np.uint8(blob4) * np.array(pink_image))
    # add slime coloring
    beauty = PIL.Image.new("RGBA", background.size)
    beauty = PIL.Image.alpha_composite(beauty, background)
    beauty = PIL.Image.alpha_composite(beauty, mask_new)
    # add bounding box
    beauty_array = np.asarray(beauty)
    beauty_array = draw_box(beauty_array, x_center, y_center, radius)
    # optional: add inscribed circle to represent petri dish
    beauty_array = draw_circle(beauty_array, x_center, y_center, radius)
    final = PIL.Image.fromarray(beauty_array)
    # add jaccard text
    fnt = PIL.ImageFont.truetype(font="resources/arial.ttf", size=18)
    text_coords_above = (x_center - radius - 5, y_center - radius - 25)  # subtract to compensate for box thickness
    text_coords_below = (x_center - radius - 5, y_center + radius + 5)  # add to compensate for box thickness
    # place text where there is more vertical space
    if background.size[1] - (y_center + radius) >= y_center - radius:
        text_coords = text_coords_below
    else:
        text_coords = text_coords_above
    draw = PIL.ImageDraw.Draw(final)
    draw.text(text_coords, f"Predicted Jaccard: {jac:.3f}", font=fnt, fill=(191, 64, 191))  # use purple text
    final.save(result_folder + "/" + filename + "_prediction.png")
