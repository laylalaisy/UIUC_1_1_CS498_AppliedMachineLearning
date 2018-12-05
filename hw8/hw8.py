import numpy as np
from scipy import misc
from numpy import linalg as la
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import sys
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":
    K = 10  # number of segments
    filename = "2cf92c40c3f3306321d789f7e9c12893.jpg"

    # read in image
    img = misc.imread(filename)
    width, length = img.shape[0:2]
    # process data of image
    # transpose: color(3) * width * length
    img = img.transpose(2, 0, 1)
    # reshape: color(3) * (width * length) = color * pixels
    img = img.reshape(3, -1).astype(float)
    img = img.transpose()

    # number of pixels
    N = img.shape[0]
    # number of features/ number of colors
    D = img.shape[1]

    gmm = GaussianMixture(n_components=K).fit(img)
    model = gmm.predict(img)
    mu = gmm.means_

    res = mu[model] * 255

    res = np.array(res).reshape(width, length, D)
    misc.imsave('4_20.png', res)