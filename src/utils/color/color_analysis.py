import logging
import numpy as np
from src.utils.color.color_conversion import ciede2000_distance
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)

class ColorMetricCalculator:
    """
    Calculates color differences (Delta E 2000) against a set target palette.

    This class is initialized with a "ground truth" list of CIELAB colors
    (the target palette, usually from the reference image). It then provides
    methods to compare new lists of segmented colors against this target palette.
    """
    def __init__(self, target_colors_lab: np.ndarray):
        """
        Initializes the calculator with target colors in LAB format
        Args:
            target_colors_lab: NumPy array of target colors in CIELAB
                shape (n_targets, 3) 
        """
        if not isinstance(target_colors_lab, np.ndarray) or target_colors_lab.ndim != 2 or target_colors_lab.shape[1] != 3:
            raise ValueError("target_colors_lab must be a NumPy array with shape (n, 3)")
        self.target_colors_lab = target_colors_lab
        if target_colors_lab.shape[0] == 0:
             logger.warning("ColorMetricCalculator initialized with zero target colors.")

    def compute_similarity(self, segmented_colors_lab: Union[List, np.ndarray]) -> np.ndarray:
        """
        Computes minimum CIEDE2000 distance for each segmented LAB color to the nearest target LAB color.
        Lower distance means higher similarity.

        Args:
            segmented_colors_lab (Union[List, np.ndarray]): List or array of segmented colors in CIELAB.

        Returns:
            List[float]: List of minimum distances for each segmented color. Empty list if inputs are invalid.
        """
        if segmented_colors_lab is None or len(segmented_colors_lab) == 0:
            logger.warning("No segmented LAB colors provided for similarity calculation.")
            return []
        if self.target_colors_lab.shape[0] == 0:
             logger.warning("No target colors available for similarity calculation.")
             # Return infinity for all segmented colors as no match is possible
             return [float('inf')] * len(segmented_colors_lab)
             
        min_distances = []
        for color_lab in segmented_colors_lab:
            try:
                # Calculate distance to all target colors and find the minimum
                distances = [ciede2000_distance(color_lab, target) for target in self.target_colors_lab]
                min_distances.append(min(distances))
            except Exception as e:
                 logger.error(f"Error computing similarity for color {color_lab}: {e}")
                 min_distances.append(float('inf')) # Append infinity on error
                 
        return min_distances
    
    def find_best_matches(self, segmented_colors_lab: Union[List, np.ndarray]) -> List[Tuple[int, int, float]]:
        """
        Finds the best matching target color index for each segmented LAB color
        
        Args:
            segmented_colors_lab(Union[List, np.ndarry]):  Lİst or array of segmented colors in CIELAB
            
        Returns:
            List[Tuple[int, int, float]]: List of tuples: (segmented_color_index, best_target_color_index, min_distance).
                                           best_target_color_index is -1 if no target colors exist or error occurs.
                                           min_distance is inf if no target colors or error.
        """
        if segmented_colors_lab is None or len(segmented_colors_lab) == 0:
            logger.warning("No segmented LAB colors provided for finding best matches")
            return []
        
        best_matches = []
        if self.target_colors_lab_shape[0] == 0:
            logger.warning("No target colors available for matching")
            # return matches with -1 index and infinite distance
            return [(i, -1, float('inf')) for i in range(len(segmented_colors_lab))]
        
        for i, color_lab in enumerate(segmented_colors_lab):
            min_distance = float('inf')
            best_target_idx = -1 # default to -1 means no match
            try:
                for j, target_lab in enumerate(self.target_colors_lab):
                    distance = ciede2000_distance(color_lab, target_lab)
                    if distance < min_distance:
                        min_distance = distance
                        best_target_idx = j
                
                best_matches.append((i, best_target_idx, min_distance))
            except Exception as e:
                 logger.error(f"Error finding best match for segmented color index {i} ({color_lab}): {e}")
                 best_matches.append((i, -1, float('inf'))) # Append error state

        return best_matches
    
    def compute_delta_e(self,
                        segmented_lab_colors: Union[List, np.ndarray],
                        best_matches: List[Tuple[int, int, float]]) -> float:
        """
        Computes the mean Delta E (CIEDE2000) between the segmented LAB colors and their best matching target LAB colors
        
        Args: 
            segmented_lab_colors (Union[List, np.ndarray]): the list/ array of segmented colors in CIELAB
            best_matches (List[Tuple[int, int, float]]): the output from find_best_matches
            
        Returns:
            Float: the mean CIEDE2000 distance or float('nan') if no valid matches found        
        """
        if not segmented_lab_colors or not best_matches or self.target_colors_lab.shape[0] == 0:
            return float('nan')
        
        distances = []
        for seg_idx, target_idx, distance in best_matches:
            # Check if the match is valid (target_idx != -1 and distance is not inf)
            if target_idx != -1 and distance != float('inf'):
                 # Optional additional check: ensure indices are within bounds
                 if 0 <= seg_idx < len(segmented_lab_colors) and 0 <= target_idx < len(self.target_colors_lab):
                     distances.append(distance)
                 else:
                      logger.warning(f"Match indices out of bounds: seg_idx={seg_idx}, target_idx={target_idx}. Skipping.")
                      
        return np.mean(distances) if distances else float('nan')
    
    def compute_all_delta_e(self, segmented_lab_colors: Union[List, np.ndarray]) -> List[float]:
        """
        Computes the minimum Delta E (CIEDE2000) for EACH segmented LAB color 
        against ALL target LAB colors. This is equivalent to compute_similarity.

        Args:
            segmented_lab_colors (Union[List, np.ndarray]): List or array of segmented colors in CIELAB.

        Returns:
            List[float]: List containing the minimum CIEDE2000 distance for each 
                         segmented color to its nearest target color. 
                         Returns empty list on error or invalid input.
        """
        # This function essentially does the same as compute_similarity
        # We can just call compute_similarity for clarity and reuse
        return self.compute_similarity(segmented_lab_colors)    