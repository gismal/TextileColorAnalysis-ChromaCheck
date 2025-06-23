import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

def ciede2000_distance(color1, color2):
    color1_lab = np.array(color1).reshape((1, 1, 3))
    color2_lab = np.array(color2).reshape((1, 1, 3))
    return deltaE_ciede2000(color1_lab, color2_lab)[0][0]

def visualize_color_comparison(color1, color2):
    # Calculate Delta E
    delta_e = ciede2000_distance(color1, color2)

    # Create a plot to visualize the colors
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # Display the first color
    color1_rgb = np.uint8([[color1]])
    ax[0].imshow(color1_rgb)
    ax[0].set_title(f'Color 1: {color1}')
    ax[0].axis('off')

    # Display the second color
    color2_rgb = np.uint8([[color2]])
    ax[1].imshow(color2_rgb)
    ax[1].set_title(f'Color 2: {color2}')
    ax[1].axis('off')

    # Add a super title with the Delta E value
    plt.suptitle(f'Delta E (CIEDE2000): {delta_e:.2f}', fontsize=16)
    plt.show()

# Example colors
color1 = [0, 0, 128]
color2 = [0, 0, 128]

# Visualize the comparison
visualize_color_comparison(color1, color2)
