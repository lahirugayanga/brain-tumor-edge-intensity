import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io

median_data = []
files = glob.glob("D:/EdgeIntensity/Data/TumorImage/*.jpg")
for myFile in files:
    print(myFile)
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    median_data.append(image)

# EDGE DETECTION---------------------------------------------------------------------------------
tumor_mask = []
files = glob.glob("D:/EdgeIntensity/Data/TumorMask/*.jpg")
for myFile in files:
    print(myFile)
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    tumor_mask.append(image)

length = len(tumor_mask)
canny_edge_data = []

for i in range(length):
    L1 = cv2.Canny(tumor_mask[i], 100, 200, L2gradient=False)
    io.imshow(L1, cmap='gray')
    io.show()
    canny_edge_data.append(L1)

# EDGE DILATION-----------------------------------------------------------------------------------
edge_dilated_data = []

for i in range(length):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(canny_edge_data[i], kernel, iterations=8)
    io.imshow(dilation, cmap='gray')
    io.show()
    edge_dilated_data.append(dilation)

# MASKING-----------------------------------------------------------------------------------------
masked_data = []

for i in range(length):
    edge_dilated_data[i] = cv2.resize(edge_dilated_data[i], median_data[i].shape[1::-1])
    dst = cv2.bitwise_and(median_data[i], edge_dilated_data[i])
    io.imshow(dst, cmap='gray')
    io.show()
    masked_data.append(dst)

# INTENSITY--------------------------------------------------------------------------------------
histogram = 0
for i in range(length):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    ix = ndimage.filters.convolve(masked_data[i], kx)
    iy = ndimage.filters.convolve(masked_data[i], ky)
    g = np.hypot(ix, iy)
    g = g / g.max() * 255
    histogram = cv2.calcHist([g.astype(np.uint8)], [0], None, [256], [128, 256])
    list_data = np.array(g, dtype=np.float64)
    std_deviation = np.std(list_data)
    print("Edge intensity = " + str(std_deviation))
    if std_deviation <= 43.5:
        print("Edge nature = Sharp edge")
        x = "Sharp Edge"
    else:
        print("Edge nature = Cloudy Edge")
        x = "Cloudy Edge"
plt.plot(histogram)
plt.show()
