from mrf_3d import MRF, SecondOrderMRF
from sklearn.mixture import GaussianMixture
from gmm import GaussianMixtureModel
# from matplotlib import pyplot as plt
import numpy as np
from resegment import ShapeSegmentMRF, resegment
from shapes import ShapeCollection
from image import Image
from shape_fitting import get_all_shapes
from skimage import io, img_as_float#, color
from scipy import misc
from subprocess import call
import os

class MRFScripts():
    def __init__(self,gmm_switch=True,verbose=False):
        self.verbose = verbose
        self.img_viewer_path = r"C:\Program Files\JPEGView64\JPEGView.exe" # r"C:\bin\JPEGView64\JPEGView.exe"
        self.last_fname = None
        self.last_img = None
        self.last_gmm = None
        self.last_k = None
        if gmm_switch:
            self.gmm_type = 'sklearn'
        else:
            self.gmm_type = 'own'

    def crop_image(self,filename):
        img = io.imread(filename)
        img = img[:,180:-180,:]
        return img

    def get_img(self,fname,ppr=True):
        if self.last_fname == fname and self.last_img is not None:
            return self.last_img
        img_data = self.crop_image(fname)
        new_img = Image(data=img_data,pepper=ppr,scale=10)
        self.last_img = new_img
        return new_img

    def get_gmm(self,fname,K):
        # first step of every mrf procedure
        if self.last_fname == fname and self.last_gmm is not None and self.last_k == K:
            return self.last_gmm
        test_img = self.get_img(fname)
        if self.gmm_type == 'sklearn':
            test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=self.verbose)
            d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0] * test_img._data.shape[1], test_img._data.shape[2]))
            new_gmm = test_gmm.fit(d2_array)
        else:
            test_gmm = GaussianMixtureModel(test_img,K,verbose=self.verbose)
            new_gmm = test_gmm.estimate_parameters()
        self.last_fname = fname
        self.last_gmm = new_gmm
        self.last_k = K
        return new_gmm

    def gmm(self,fname, K):
        print('good ol\' gmm')
        test_img = self.get_img(fname)
        test_gmm = self.get_gmm(fname,K)
        if self.gmm_type == 'sklearn':
            test_mrf = MRF(test_img, test_gmm.means_, [np.eye(3) * test_gmm.covariances_[i] for i in range(K)],verbose=self.verbose)
        else:
            test_mrf = MRF(test_img, test_gmm[1], test_gmm[2],verbose=self.verbose)
        savename = '{}_gmm.png'.format(fname[:-4])
        io.imsave(savename,misc.imresize(test_mrf.labels*255,15.0))
        call([self.img_viewer_path,os.path.abspath(savename)])

    def vanilla_MRF(self,fname, K):
        print('basic mrf')
        test_img = self.get_img(fname)
        test_gmm = self.get_gmm(fname,K)
        if self.gmm_type == 'sklearn':
            test_mrf = MRF(test_img, test_gmm.means_, [np.eye(3) * test_gmm.covariances_[i] for i in range(K)],verbose=self.verbose)
        else:
            test_mrf = MRF(test_img, test_gmm[1], test_gmm[2],verbose=self.verbose)
        savename = '{}_vanillaMRF.png'.format(fname[:-4])
        test_mrf.icm()
        io.imsave(savename,misc.imresize(test_mrf.labels*255,15.0))
        call([self.img_viewer_path,os.path.abspath(savename)])

    def hard_boundary_MRF(self,fname,K):
        print('hard boundary mrf')
        test_img = self.get_img(fname)
        test_gmm = self.get_gmm(fname,K)
        if self.gmm_type == 'sklearn':
            test_mrf = SecondOrderMRF(test_img, test_gmm.means_, [np.eye(3) * test_gmm.covariances_[i] for i in range(K)],verbose=self.verbose)
        else:
            test_mrf = SecondOrderMRF(test_img, test_gmm[1], test_gmm[2],verbose=self.verbose)
        test_mrf.icm()
        # plt.imshow(test_mrf.labels)
        # plt.savefig('hard_boundary_MRF.png')
        savename = '{}_hard_boundary_MRF.png'.format(fname[:-4])
        io.imsave(savename,misc.imresize(test_mrf.labels*255,15.0))
        call([self.img_viewer_path ,os.path.abspath(savename)])


    def segment_with_priors(self,fname,K):
        test_img = self.get_img(fname,False)
        test_gmm = self.get_gmm(fname,K)
        if self.gmm_type == 'sklearn':
            test_mrf = SecondOrderMRF(test_img, test_gmm.means_, [np.eye(3) * test_gmm.covariances_[i] for i in range(K)],verbose=self.verbose)
        else:
            test_mrf = SecondOrderMRF(test_img, test_gmm[1], test_gmm[2],verbose=self.verbose)
        test_mrf.icm()
        im = np.zeros([test_img.height, test_img.width, 3])
        for i in range(test_img.height):
            for j in range(test_img.width):
                im[i, j] = test_mrf.means[test_mrf.labels[i, j]]
        all_shapes = get_all_shapes(test_mrf.labels)
        for shape in all_shapes:  # [all_shapes[i] for i in range(len(all_shapes)) if i!=8]:
            # shape = all_shapes[6]
            mask = shape.get_mask(im.shape[0], im.shape[1])
            im = resegment(test_mrf,mask, im, shape.label, test_mrf.means, test_mrf.variances)
        # plt.imshow(im)
        # plt.savefig('total_resegment.png')
        savename = '{}_total_resegment.png'.format(fname[:-4])
        io.imsave(savename,misc.imresize(im,15.0))
        call([self.img_viewer_path,os.path.abspath(savename)])

       
        


if __name__ == '__main__':
    test_mrr = MRFScripts()
    K=8
    test_img = './scrot/2.png'
    test_mrr.gmm(test_img,K)
    # test_mrr.vanilla_MRF(test_img,K)
    # hard_boundary_MRF(test_img,K)
    # segment_with_priors(test_img,K)



