from skimage import measure, ndimage
from scipy import ndimage
import math
import numpy as np

def get_bounds(examine):
	pot_h, pot_w = max(np.sum(examine,axis=0)), max(np.sum(examine,axis=1))
	cntr = ndimage.measurements.center_of_mass(examine)
	return cntr, pot_h, pot_w

# input: (potentially) binary mask from a label (bin_mask)
def get_shapes_of_regions(bin_mask,bg_thresh=.75,rect_ellipse_thresh=.5,too_small_thresh=200):
	# this does connected components, i think implementing this algo would be outside of the scope
	# of this class because it's not probabilistic.. (if u look @ source you'll see it's all c code anyways
	# so reimplementing will just slow our code down even more)
	labeling, nk = measure.label(ndimage.morphology.binary_closing(bin_mask),return_num=True)
	tor = []
	if nk == 0: return tor
	# check that this isnt background -.-
	_, pot_h, pot_w = get_bounds(bin_mask)
	tot_h, tot_w = bin_mask.shape[0], bin_mask.shape[1]
	if all([pot_h/tot_h >= bg_thresh, pot_w/tot_w >= bg_thresh]): return tor
	# get rid of things that are not major shapes..
	tru_ks = [k for k in range(1,nk+1) if np.sum(labeling==k) > too_small_thresh] 
	
	for k in tru_ks:

		examine = labeling == k
		cntr, pot_h, pot_w = get_bounds(examine)
		bot, top = max(0,math.floor(cntr[0]-pot_h/2)),min(math.floor(cntr[0]+pot_h/2),examine.shape[0])
		left, right = max(0,math.floor(cntr[1]-pot_w/2)),min(math.floor(cntr[1]+pot_w/2),examine.shape[1])

		x, y = np.meshgrid(np.arange(examine.shape[1]), np.arange(examine.shape[0]))
		x -= math.floor(cntr[1])
		y -= math.floor(cntr[0])
		elliptical_mask = ((x * x)/(pot_w**2)*4 + (y * y)/(pot_h**2)*4 < 1)

		bounded_box = examine[bot:top,left:right]
		bounded_oval = (examine*elliptical_mask)[bot:top,left:right]
		# print('shape',k)
		if np.sum(bounded_box^bounded_oval)/ (pot_w*pot_h - np.sum(elliptical_mask)) > rect_ellipse_thresh:
			shape_type = 'Rectangle'
		else:
			shape_type = 'Ellipse'
		tor.append(random_shape_gen.Shape(left,top,right,bot,shape_type))
	return tor

def get_all_shapes(current_labeling):
	s_tor = []
	# if you would like to plot stuff uncomment the commented out lines..

	# to_display = np.zeros((current_labeling.shape[0],current_labeling.shape[1]))

	for label in range(0,np.max(current_labeling)+1):
	    dub = test_mrf.labels == label
	    ss = get_shapes_of_regions(dub)
	    if len(ss) > 0:
	        for s in ss:
	        	s_tor.append(s)
	        	# to_display += s.get_mask(current_labeling.shape[0],current_labeling.shape[1])
	return s_tor#, to_display

if __name__ == '__main__':
	from draw import random_shape_gen # i have shape_gen inside a draw folder idk code will need refactoring tomorrow
	from mrf_3d import MRF, Image

	from sklearn.mixture import GaussianMixture # heh
	test_file = './draw/scrot/0.png'
	test_img = Image(filename=test_file,pepper=True,scale=1) # added some preprocessing params to image
	test_truth = random_shape_gen.MultiShapeHolder(test_img.width,test_img.height)
	test_truth.get_truth('./draw/scrot/0.txt')
	k = len(test_truth.shapes)
	test_gmm = GaussianMixture(k+1,'full',init_params='random',verbose=1) # k+1 because of background!!
	d2_array = np.reshape(np.ravel(test_img._data),(test_img.height*test_img._data.width,test_img.bitdepth)) # flatten
	test_gmm.fit(d2_array)
	test_mrf = MRF(test_img,test_gmm.means_,test_gmm.covariances_,verbose=True)
	test_mrf.icm()

