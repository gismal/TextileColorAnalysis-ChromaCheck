import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

def convert_colors_to_cielab(avg_colors):
    avg_colors_lab = []
    for color in avg_colors:
        color_bgr = np.uint8([[color]])
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    return avg_colors_lab

def ciede2000_distance(color1, color2):
    color1_lab = np.array(color1).reshape((1, 1, 3))
    color2_lab = np.array(color2).reshape((1, 1, 3))
    return deltaE_ciede2000(color1_lab, color2_lab)[0][0]

def calculate_similarity(segmentation_data, target_colors):
    similarity_scores = []
    avg_colors_lab = segmentation_data['avg_colors_lab']
    for color_lab in avg_colors_lab:
        if np.all(np.array(color_lab) <= np.array([5, 130, 130])):  # Check for nearly black segment
            similarity_scores.append(0)
            continue
        min_distance = float('inf')
        for target_color in target_colors:
            distance = ciede2000_distance(color_lab, target_color)
            if distance < min_distance:
                min_distance = distance
        similarity_score = max(0, 100 - min_distance)
        similarity_scores.append(similarity_score)
    return similarity_scores

def find_best_matches(segmentation_data, reference_segmentation_data):
    avg_colors_lab = segmentation_data['avg_colors_lab']
    ref_avg_colors_lab = reference_segmentation_data['avg_colors_lab']

    best_matches = []
    for i, color_lab in enumerate(avg_colors_lab):
        if np.all(np.array(color_lab) <= np.array([5, 130, 130])):  # Ignore nearly black segments
            best_matches.append((i, -1))
            continue
        min_distance = float('inf')
        best_match_idx = -1
        for j, ref_color_lab in enumerate(ref_avg_colors_lab):
            distance = ciede2000_distance(color_lab, ref_color_lab)
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        best_matches.append((i, best_match_idx))

    return best_matches
