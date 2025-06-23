import cv2
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def visualize_keypoints(img, keypoints, title, max_display=500):
        try:
            keypoints = keypoints[:max_display]
            img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            plt.title(title + "\n(Keypoints detected in the image)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error visualizing keypoints: {e}")

    @staticmethod
    def visualize_matches(img1, kp1, img2, kp2, matches, title):
        try:
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(20, 10))
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title(title + "\n(Matching keypoints between images)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error visualizing matches: {e}")

    @staticmethod
    def visualize_segments(segments, title):
        try:
            fig, axes = plt.subplots(1, len(segments), figsize=(20, 5))
            for ax, segment in zip(axes, segments):
                ax.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
                ax.axis('off')
            plt.suptitle(title + "\n(Segments of the image for color analysis)")
            plt.show()
        except Exception as e:
            print(f"Error visualizing segments: {e}")

    @staticmethod
    def visualize_pattern_detection(img, loc, template_size, title):
        try:
            img_copy = img.copy()
            if len(loc[0]) > 0:
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img_copy, pt, (pt[0] + template_size[0], pt[1] + template_size[1]), (0, 255, 0), 2)
            else:
                print("No patterns detected.")
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.title(title + "\n(Rectangles show detected patterns)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error visualizing pattern detection: {e}")

    @staticmethod
    def visualize_images(images, titles):
        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        plt.show()