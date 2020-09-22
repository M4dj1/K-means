import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Kmeans(imgMat, k):
    size = imgMat.itemsize                                   # length of each element of array in bytes (Needed in strides)
    height, length = imgMat.shape[:2]                        # retrieving picture's dimensions
    distancesOld = np.zeros((height, length,k))              # initializing Distances
    imgRes = imgMat.copy()                                   # "Results" image
    centroids = np.random.randint(0, 256, size=(1, k))       # initializing centroids
    img_view = np.lib.stride_tricks.as_strided(imgMat, shape=(height, length, 1, k),    # the shape of the array after striding
            strides=(length*size, 1*size, 0*size, 0*size))   # stride size mesured in bytes (size=1byte) step : full length then one step
    while True:                                              # loop in num of iterations
        distances = np.array(np.abs(img_view - centroids)[:, :, 0, :])   # Manhattan Distance calculating
        if (np.array_equal(distancesOld,distances)) :        # test if converged
            break                                            # break if (new distances == old distances)
        idx = np.argmin(distances, axis=-1)                  # indices of minimum distances
        for i in range(k):                                   # loop in centroids
            imgRes[idx == i] = centroids[0][i]               # if indice of minimum distance equals to cluster's n then assign the indice with centroid value
            cluster = imgMat[idx == i]                       # group the minimum distances indices in one cluster for every centroid's cluster
            if len(cluster) == 0:                            # if the cluster is empty there is no need to update the centroid
                continue                                     # iterate next element
            centroids[0][i] = np.mean(cluster)               # update centroid with mean(cluster)
        distancesOld = distances                             # updating old distances
    return imgRes

img = Image.open('bird.jpg').convert('L')
imgMat = np.array(img)
k = int(input('Enter k clusters (classes) : '))
imgRes = Kmeans(imgMat, k)

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Simplified Image')
plt.imshow(Image.fromarray(imgRes), cmap='gray')
plt.show()
