import cv2
import numpy as np


def detectChars(img):
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)

    # detect bakground pixel
    pixel_counts = {}

    for i in range(height):
        if gray.item(i, 0) in pixel_counts:
            pixel_counts[gray.item(i, 0)] += 1
        else:
            pixel_counts[gray.item(i, 0)] = 1

    max_count = 0
    bg_pixel = 0
    for key in pixel_counts.keys():
        if pixel_counts[key] > max_count:
            max_count = pixel_counts[key]
            bg_pixel = key

    letter_mode = 0
    right_edge = 0
    update_cut = 0
    cuts = [0]
    for j in range(width):
        if letter_mode == 0:
            count = 0
            for i in range(height):
                if gray.item(i, j) != bg_pixel:
                    count += 1

                if count > 40:
                    letter_mode = 1
                    left_edge = j
                    # print(left_edge)

                    if right_edge != 0:
                        if update_cut != 0:
                            update_cut = 0
                            if len(cuts) != 0:
                                cuts[len(cuts) - 1] = (
                                    cuts[len(cuts) - 1]
                                    + (right_edge + left_edge) / 2
                                ) / 2
                            break

                        cuts.append((right_edge + left_edge) / 2)
                    break

        if letter_mode != 0:
            count = 0
            for i in range(height):
                if gray.item(i, j) != bg_pixel:
                    count += 1

            if count < 30:
                letter_mode = 0
                right_edge = j
                # print(right_edge)

                if right_edge - left_edge < 15:
                    update_cut = 1

    cuts.append(width)

    letters = []

    for i in range(len(cuts) - 1):
        letters.append(img[0:height, int(cuts[i]):int(cuts[i + 1]), :])

    return letters
