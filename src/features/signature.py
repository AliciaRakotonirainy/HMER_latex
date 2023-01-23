import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import similaritymeasures
rcParams['text.usetex'] = True

# Read in the image from disk.
gray_image = cv2.imread("HMER_latex/data/CROHME_train_2011_PNG/formulaire001-equation001.png", cv2.IMREAD_GRAYSCALE)

# Display the image
cv2.imshow("Original Grayscale Image", gray_image)
cv2.waitKey(0)

gray_image = gray_image[:,:200]
cv2.imshow("Original Grayscale Image", gray_image)
cv2.waitKey(0)



# Binarize the image
ret, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display the image
cv2.imshow("Mask Image", mask)
cv2.waitKey(0)

# Extract largest blob
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

largest_contour = np.squeeze(largest_contour)


## compute centroid
M = cv2.moments(gray_image)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

plt.imshow(gray_image, cmap="gray")
plt.scatter(largest_contour[:,0], largest_contour[:,1])
plt.scatter(cX, cY, c="red")
plt.show()

## compute distance from centroid
x = largest_contour[:, 0]
y = largest_contour[:, 1]
distances = np.sqrt((x - cX) ** 2 + (y - cY) ** 2)

# scale distances so that the maximal distance is 1
distances = distances / distances.max()

# angles of each 
angles = np.degrees(np.arctan2(y - cY, x - cX))

plt.plot(angles, distances)
plt.show()




# Write the reference symbol on image
fig, ax = plt.subplots(figsize=(1.1,1.1))
ax.text(0,0, r"$\phi$", ha='center', size=70)
ax.axis('off')
fig.tight_layout()
plt.savefig('HMER_latex/data/references/phi.png')

ref = cv2.imread('HMER_latex/data/references/phi.png', cv2.IMREAD_GRAYSCALE)

# rescale reference to the size of the character
ref = cv2.resize(ref, (gray_image.shape[1],gray_image.shape[0]))

cv2.imshow("Original Grayscale Image", ref)
cv2.waitKey(0)


# Binarize the image
ret, mask_ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display the image
cv2.imshow("Mask Image", mask_ref)
cv2.waitKey(0)

# Extract largest blob
contours_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour_ref = max(contours_ref, key=cv2.contourArea)

largest_contour_ref = np.squeeze(largest_contour_ref)


## compute centroid
M_ref = cv2.moments(ref)
cX_ref = int(M_ref["m10"] / M_ref["m00"])
cY_ref = int(M_ref["m01"] / M_ref["m00"])

plt.imshow(ref, cmap="gray")
plt.scatter(largest_contour_ref[:,0], largest_contour_ref[:,1])
plt.scatter(cX_ref, cY_ref, c="red")
plt.show()

## compute distance from centroid
x_ref = largest_contour_ref[:, 0]
y_ref = largest_contour_ref[:, 1]
distances_ref = np.sqrt((x_ref - cX_ref) ** 2 + (y_ref - cY_ref) ** 2)

# scale distances
distances_ref = distances_ref / distances_ref.max()

# angles of each 
angles_ref = np.degrees(np.arctan2(y_ref - cY_ref, x_ref - cX_ref))

plt.scatter(angles_ref, distances_ref, s=1, c="blue", label = "Signature of reference symbol")
plt.scatter(angles, distances, s = 1, c="red", label="Signature of handwritten character")
plt.legend()
plt.show()


def make_outer_contour(angles, distances, step=3):
    for i in range(-180,180,step):
        idx_similar_angle = np.where((i <= angles) & (angles < i + step))[0]

        if len(idx_similar_angle) > 0:
            max_distance = np.max(distances[idx_similar_angle])
            distances[idx_similar_angle] = max_distance

    return(angles, distances)


angles, distances = make_outer_contour(angles, distances)
angles_ref, distances_ref = make_outer_contour(angles_ref, distances_ref)

plt.scatter(angles_ref, distances_ref, s=1, c="blue", label = "Signature of reference symbol")
plt.scatter(angles, distances, s = 1, c="red", label="Signature of handwritten character")
plt.legend()
plt.show()







## Attention ! Ce ne sont pas forcément des courbes de distribution puisqu'on peut avoir plusieurs valeurs pour le même angle !

angles = angles.reshape(-1, 1)
distances = distances.reshape(-1,1)
signature = np.concatenate((angles, distances), axis = 1)

angles_ref = angles_ref.reshape(-1,1)
distances_ref = distances_ref.reshape(-1,1)
signature_ref = np.concatenate((angles_ref, distances_ref), axis = 1)

# the higher the distance, the most different the 2 symbols are
pcm = similaritymeasures.pcm(signature, signature_ref)
df = similaritymeasures.frechet_dist(signature, signature_ref)












