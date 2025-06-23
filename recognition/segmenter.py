class Segmenter:
    @staticmethod
    def adaptive_segment_image(image, max_segments=5):
        if image is None:
            raise ValueError("Invalid image provided for segmentation.")
        try:
            height, width = image.shape[:2]
            num_segments = min(max_segments, width // 100)
            segment_width = width // num_segments
            segments = []
            for j in range(num_segments):
                segment = image[:, j * segment_width:(j + 1) * segment_width]
                segments.append(segment)
            return segments
        except Exception as e:
            print(f"Error during adaptive segmentation: {e}")
            return []
