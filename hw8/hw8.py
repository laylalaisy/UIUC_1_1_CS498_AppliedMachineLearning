import numpy as np
from scipy import misc
from sklearn.mixture import GaussianMixture

# scale data between [0, 1]
def scale_data(img, D, N):
    for i in range(D):
        Max = max(img[i, :])
        Min = min(img[i, :])
        Scale = Max - Min
        for j in range(N):
            img[i, j] = 1.0 * (img[i, j] - Min) / Scale
    return img



if __name__ == "__main__":
    # read in image
    img = misc.imread("2cf92c40c3f3306321d789f7e9c12893.jpg")

    # process data of image
    # transpose: color(3) * width * length
    img = img.transpose(2, 0, 1)
    # reshape: color(3) * (width * length) = color * pixels
    img = img.reshape(3, -1).astype(float)

    # number of pixels
    N = img.shape[1]
    # number of features/ number of colors
    D = 3
    # number of segments
    K = 20

    img = scale_data(img, D, N)









