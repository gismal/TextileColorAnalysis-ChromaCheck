import logging
import numpy as np
from src.utils.image_utils import ciede2000_distance

class ColorMetricCalculator:
    def __init__(self, target_colors):
        self.target_colors = target_colors

    def compute_similarity(self, segmented_colors):
        """Compute similarity scores between segmented and target colors."""
        if not segmented_colors:
            logging.warning("No segmented colors provided.")
            return []
        similarities = []
        for color in segmented_colors:
            min_distance = min(ciede2000_distance(color, target) for target in self.target_colors)
            similarities.append(min_distance)
        return similarities

    def find_best_matches(self, segmented_colors):
        """Find best matches between segmented and target colors."""
        if self.target_colors.shape[0] == 0:
            logging.error("No target colors provided.")
            return []
        if not segmented_colors:
            logging.warning("No segmented colors provided.")
            return []
        best_matches = []
        for i, color in enumerate(segmented_colors):
            if np.all(np.array(color) <= np.array([5, 130, 130])):  # Skip near-black colors
                best_matches.append((i, -1, float('inf')))
                continue
            min_distance = float('inf')
            best_target_idx = -1
            for j, target in enumerate(self.target_colors):
                distance = ciede2000_distance(color, target)
                if distance < min_distance:
                    min_distance = distance
                    best_target_idx = j
            best_matches.append((i, best_target_idx, min_distance))
        return best_matches
    
    def compute_delta_e(self, segmented_colors, lab_converter, best_matches):
            """Compute mean Delta E for segmented colors using best matches and a LAB converter."""
            if not segmented_colors or not best_matches:
                return float('nan')
            distances = []
            for test_idx, ref_idx, _ in best_matches:
                if 0 <= test_idx < len(segmented_colors) and ref_idx != -1 and 0 <= ref_idx < len(self.target_colors):
                    lab_color = lab_converter(segmented_colors[test_idx])
                    target_color = self.target_colors[ref_idx]
                    distances.append(ciede2000_distance(lab_color, target_color))
            return np.mean(distances) if distances else float('nan')