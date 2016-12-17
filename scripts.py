from mrf_3d import MRF, SecondOrderMRF
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import numpy as np
from resegment import ShapeSegmentMRF
from shapes import ShapeCollection
from image import Image
from resegment import ShapeSegmentMRF, resegment
from shape_fitting import get_all_shapes
def gmm(image, K):
    test_img = Image(filename=image,scale = 2, pepper=False)
    test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=1)
    d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0] * test_img._data.shape[1], test_img._data.shape[2]))
    res = test_gmm.fit(d2_array)
    test_mrf = MRF(test_img, res.means_, [np.eye(3) * res.covariances_[i] for i in range(K)])
    plt.imshow(test_mrf.labels)
    plt.savefig('gmm.png')

def vanilla_MRF(image, K):
    test_img = Image(filename=image, scale=2, pepper=True)
    test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=1)
    d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0] * test_img._data.shape[1], test_img._data.shape[2]))
    res = test_gmm.fit(d2_array)
    test_mrf = MRF(test_img, res.means_, [np.eye(3) * res.covariances_[i] for i in range(K)])
    test_mrf.icm()
    plt.imshow(test_mrf.labels)
    plt.savefig('vanillaMRF.png')

def hard_boundary_MRF(image, K):
    test_img = Image(filename=image, scale=2, pepper=True)
    test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=1)
    d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0] * test_img._data.shape[1], test_img._data.shape[2]))
    res = test_gmm.fit(d2_array)
    test_mrf = SecondOrderMRF(test_img, res.means_, [np.eye(3) * res.covariances_[i] for i in range(K)])
    test_mrf.icm()
    im = np.zeros([test_img.height, test_img.width, 3])
    for i in range(test_img.height):
        for j in range(test_img.width):
            im[i, j] = test_mrf.means[test_mrf.labels[i, j]]
    plt.imshow(im)
    plt.savefig('hard_boundary_MRF.png')

def segment_with_priors(image, K):
    test_img = Image(filename=image, scale=2, pepper=True)
    print(test_img[10,10])
    test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=1)
    d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0] * test_img._data.shape[1], test_img._data.shape[2]))
    res = test_gmm.fit(d2_array)
    # print(res.means_)
    test_mrf = SecondOrderMRF(test_img, res.means_, [np.eye(3) * res.covariances_[i] for i in range(K)])
    test_mrf.icm()
    print(test_mrf.means)
    # plt.imshow(test_mrf.labels)
    # plt.show()
    im = np.zeros([test_img.height, test_img.width, 3])
    for i in range(test_img.height):
        for j in range(test_img.width):
            im[i, j] = test_mrf.means[test_mrf.labels[i, j]]
    all_shapes = get_all_shapes(test_mrf.labels)
    for shape in all_shapes:  # [all_shapes[i] for i in range(len(all_shapes)) if i!=8]:
        # shape = all_shapes[6]
        mask = shape.get_mask(62, 100)
        im = resegment(test_mrf,mask, im, shape.label, test_mrf.means, test_mrf.variances)
        # plt.imshow(mask)
        # plt.show()
        # plt.imshow(im)
        # plt.show()
    plt.imshow(im)
    plt.savefig('total_resegment.png')

if __name__ == '__main__':
    K=8
    test_img = './scrot/2.png'
    # gmm(test_img,K)
    # vanilla_MRF(test_img,K)
    hard_boundary_MRF(test_img,K)
    # segment_with_priors(test_img,K)



