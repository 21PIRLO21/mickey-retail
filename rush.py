import os

import cv2
import numpy as np
from PIL import Image
import copy

def ORB_features(img1, img2, outputs=None):
    orb = cv2.ORB_create()
    key_points1 = orb.detect(img1)
    key_points2 = orb.detect(img2)
    key_points1, des1 = orb.compute(img1, key_points1)
    key_points2, des2 = orb.compute(img2, key_points2)
    # keypoints, descriptor = orb.detectAndCompute(training_gray, None)

    kp_with_size_orien = copy.copy(img1)
    kp_without_size_orien = copy.copy(img2)
    outimg1 = cv2.drawKeypoints(
        img1, keypoints=key_points1, outImage=kp_with_size_orien,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    outimg2 = cv2.drawKeypoints(
        img2, keypoints=key_points2, outImage=kp_without_size_orien,
    )

    # outimg3 = np.hstack([outimg1, outimg2])
    cv2.imwrite(outputs[0], kp_with_size_orien)
    cv2.waitKey(0)
    cv2.imwrite(outputs[1], kp_without_size_orien)
    cv2.waitKey(0)
    # plt.subplot(121)
    # plt.title('Keypoints Without Size or Orientation')
    # plt.imshow(keyp_without_size)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # BruteForce Matcher

    # Match descriptors
    matches = bf.match(des1, des2)
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    good_matches = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_matches.append(x)

    matches = sorted(matches, key=lambda x: x.distance,)
    matches_img = cv2.drawMatches(
        kp_with_size_orien, key_points1,
        kp_without_size_orien, key_points2,
        matches[:30], kp_without_size_orien, flags = 2,
    ),
    cv2.imwrite(outputs[2], matches_img[0])
    return good_matches

def SIFT_features(img1, img2, outputs=None):
    sift = cv2.SIFT_create()
    key_points1 = sift.detect(img1)
    key_points2 = sift.detect(img2)
    key_points1, des1 = sift.compute(img1, key_points1)
    key_points2, des2 = sift.compute(img2, key_points2)
    # keypoints, descriptor = sift.detectAndCompute(training_gray, None)

    kp_with_size_orien = copy.copy(img1)
    kp_without_size_orien = copy.copy(img2)
    outimg1 = cv2.drawKeypoints(
        img1, keypoints=key_points1, outImage=kp_with_size_orien,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    outimg2 = cv2.drawKeypoints(
        img2, keypoints=key_points2, outImage=kp_without_size_orien,
    )

    # outimg3 = np.hstack([outimg1, outimg2])
    cv2.imwrite(outputs[0], kp_with_size_orien)
    cv2.waitKey(0)
    cv2.imwrite(outputs[1], kp_without_size_orien)
    cv2.waitKey(0)
    # plt.subplot(121)
    # plt.title('Keypoints Without Size or Orientation')
    # plt.imshow(keyp_without_size)

    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.match(des1, des2)
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    good_matches = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_matches.append(x)

    matches = sorted(matches, key=lambda x: x.distance,)
    matches_img = cv2.drawMatches(
        kp_with_size_orien, key_points1,
        kp_without_size_orien, key_points2,
        matches[:30], kp_without_size_orien, flags = 2,
    ),
    cv2.imwrite(outputs[2], matches_img[0])
    return good_matches


def SURF_features(img1, img2, outputs=None):
    surf = cv2.xfeatures2d.SURF_create(400)

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv2.imshow("Match Result", outimage)
    cv2.waitKey(0)


if __name__ == '__main__':
    img1_path = 'data/customized/posm/feihe/2003679501475643459_3911243067085056_version0/CgrU3GZDSOuAIMxkAE8erteKB4A232.jpg'
    img2_path = 'data/customized/posm/feihe/2003679501475643459_3911243067085056_version0/feihe_131-2830378d-da4b-4247-9d13-a448dc3f57b6-0.jpg'
    output_root = 'debug/rush/SIFT'
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_path = [
        f"{output_root}/CgrU3GZDSOuAIMxkAE8erteKB4A232.jpg",
        f"{output_root}/feihe_131-2830378d-da4b-4247-9d13-a448dc3f57b6-0.jpg",
        f"{output_root}/CgrU3GZDSOuAIMxkAE8erteKB4A232_draw_match.jpg",
   ]
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    print(f"image1 ori size: {image1.shape}")
    print(f"image2 ori size: {image2.shape}")
    scale_factor = 1
    image1 = cv2.resize(image1, [image2.shape[1] // scale_factor, image2.shape[0] // scale_factor])
    print(f"image1 resized size: {image1.shape}")
    # orb_matches = ORB_features(image1, image2, output_path)
    sift_matches = SIFT_features(image1, image2, output_path)
