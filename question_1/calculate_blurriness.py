from argparse import ArgumentParser

import cv2
import numpy as np
from skimage import restoration

LOW_THRESHOLD = 89
HIGH_THRESHOLD = 400


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--img-path', type=str,
                            help='Path to the image.')
    args = arg_parser.parse_args()
    return args


def calculate_laplacian_max(image):
    '''
    Apply Laplacian filter and return the brightest pixel
    '''
    return cv2.Laplacian(image, cv2.CV_64F).max()


def deblur_image(image):
    '''
    Deconvolve image with the Wiener algorithm.
    '''
    point_spread_function = np.ones((5, 5)) / 25
    deconvolved = restoration.wiener(image,
                                     point_spread_function,
                                     balance=1, clip=False)
    return deconvolved


def normalize(*, value, prev_bounds, new_bounds):
    '''
    Scale number from previous range to new range.
    '''
    value = HIGH_THRESHOLD if value >= HIGH_THRESHOLD else value
    prev_lower, prev_upper = prev_bounds
    new_lower, new_upper = new_bounds
    prev_delta = prev_upper - prev_lower
    new_delta = new_upper - new_lower

    return (((value - prev_lower) * new_delta ) / prev_delta) + new_lower


def calculate_blurriness(*, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise TypeError(f'Image at {img_path} does not exist!')

    laplacian_max = calculate_laplacian_max(img)

    if laplacian_max <= LOW_THRESHOLD:
        # deblur very blurry images
        deblurred_img = deblur_image(img)
        laplacian_max = calculate_laplacian_max(deblurred_img)

    if laplacian_max <= LOW_THRESHOLD:
        new_bounds = (1, 2)
        prev_bounds = (1, LOW_THRESHOLD)
    else:
        new_bounds = (2, 5)
        prev_bounds = (LOW_THRESHOLD, HIGH_THRESHOLD)

    return normalize(value=laplacian_max,
                     prev_bounds=prev_bounds,
                     new_bounds=new_bounds)


def main():
    args = get_args()
    img_path = args.img_path
    blurriness_score = calculate_blurriness(img_path=img_path)
    print(f"Image: {img_path}\nBlurriness Score: {blurriness_score:.3f}")


if __name__ == '__main__':
    main()
