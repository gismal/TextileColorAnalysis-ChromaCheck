import cv2
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt

# Specify the paths to your images
reference_image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
test_image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/2.jpg'

# Load images
reference_image = cv2.imread(reference_image_path)
test_image = cv2.imread(test_image_path)

# Check if the images were loaded successfully
if reference_image is None:
    raise FileNotFoundError(f"Failed to load reference image from path: {reference_image_path}")
if test_image is None:
    raise FileNotFoundError(f"Failed to load test image from path: {test_image_path}")

# Convert images to LAB color space
lab_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)
lab_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)

# Calculate mean LAB values for a region of interest (ROI)
roi_ref = lab_ref[50:100, 50:100]
roi_test = lab_test[50:100, 50:100]

mean_lab_ref = np.mean(roi_ref, axis=(0, 1))
mean_lab_test = np.mean(roi_test, axis=(0, 1))

# Convert mean LAB values to scalars using item()
mean_lab_ref = [mean_lab_ref[0].item(), mean_lab_ref[1].item(), mean_lab_ref[2].item()]
mean_lab_test = [mean_lab_test[0].item(), mean_lab_test[1].item(), mean_lab_test[2].item()]

# Calculate color difference
color1 = LabColor(*mean_lab_ref)
color2 = LabColor(*mean_lab_test)
delta_e = delta_e_cie2000(color1, color2)

print(f'Color difference (Delta E): {delta_e}')

# Basic pattern comparison using template matching
result = cv2.matchTemplate(test_image, reference_image, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f'Max correlation value: {max_val} at location: {max_loc}')

# Visualization of the ROI and color differences
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Reference Image')
ax[1].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Test Image')
plt.show()
