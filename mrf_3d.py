import numpy as np
import math, random
import pdb

from skimage import io, img_as_float, color
from skimage.exposure import equalize_hist
from skimage.util import random_noise

# from gmm import GaussianMixtureModel as oliver_gmm

class Image():
	"""
	holds a picture
	can either supply a filename or
	data as a numpy array
	"""
	def __init__(self, filename=None, data=None):
		assert any([filename is not None, data is not None]), "you need to supply an image file or pass a picture array"

		if filename is not None:
			self._data = io.imread(filename)
		else:
			self._data = data

		# preprocessing
		# self._data = color.rgb2lab(self._data)
		# if self._data.ndim > 2:
		# 	self._data = equalize_hist(color.rgb2gray(self._data)) # convert to grayscale
		# self._data = img_as_float(self._data) # lab already normalized??
		self._data = random_noise(self._data)

		(self.height, self.width, self.bitdepth) = self._data.shape

		self.indices = [(i,j) for i in range(self.height) for j in range(self.width)]


	def __getitem__(self, item):
	# piggyback off of numpy's array indexing
		return self._data.__getitem__(item)

		
class MRF():
	def __init__(self, image, means, variances):
		# mean, var comes from gmm
		self.image = image
		self.indices = self.image.indices
		self.means, self.variances = means, variances
		self.no_classes = len(means)

		# precalculate determinants and inverses for the variances
		self.var_dets = [np.linalg.det(variances[k]) for k in range(self.no_classes)]
		self.var_invs = [np.linalg.inv(variances[k]) for k in range(self.no_classes)]

		self.labels = self.init_labels_from_gmm()

	### initialization helper functions

	def beta(self, i, j, coeff=-1):
		# possible to overwrite this function to modify our energies
		return coeff
	
	def estimate_label(self, i, j):
		# maximum log-likelihood
		lls = [self.singleton(i,j,k) for k in range(self.no_classes)]
			   # returns index of minimum as apposed to python's min which would return minimum value
			   # we want minimum since singleton is negative log-likelihood, but want max likelihood
		return np.argmin(lls) 

	def init_labels_from_gmm(self):
		label_output = np.empty((self.image.height,self.image.width), dtype='int8')
		for i in range(self.image.height):
			for j in range(self.image.width):
				label_output[i,j] = self.estimate_label(i,j)
		return label_output

	### energy calculation functions

	def singleton(self, i, j, label):
		# this is just the negative log-likelihood
		# pdb.set_trace()
		x_mu = np.matrix(self.image[i,j] - self.means[label])
		return 0.5 * (math.log(self.var_dets[label]) + x_mu * self.var_invs[label] * x_mu.T)#.item(0) # unwrap numpy matrix
	
	def doubleton(self, i, j, label):
		energy = 0.0

		for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
			if all([i + dy < self.image.height,
					i + dy >= 0,
					j + dx < self.image.width,
					j + dx >= 0]):
				if label == self.labels[i+dy,j+dx]: 
					energy -= self.beta(i,j)
				else:
					energy += self.beta(i,j)

		return energy

	def global_energy(self):
		singletons = 0
		doubletons = 0

		for i in range(self.image.height):
			for j in range(self.image.width):
				k = self.labels[i,j]
				# pdb.set_trace()
				singletons += self.singleton(i,j,k)
				doubletons += self.doubleton(i,j,k)

		return singletons + doubletons/2

	def local_energy(self, i, j, label):
		return self.singleton(i,j,label) + self.doubleton(i,j,label)

	### estimation algos for updating labels
	# efficient graph cut

	def icm(self, thresh=0.05):
		# basically loop through everything picking minimizing labeling until "convergence"
		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0

		while delta_E > thresh and counter < 10: # threshold for convergence
			delta_E = 0
			# mix up the order the indices are visited
			random.shuffle(self.indices)

			for i, j in self.indices:
				# local_energies = [self.local_energy(i,j,k) for k in range(self.no_classes)]
				self.labels[i,j] = self.estimate_label(i,j)#np.argmin(local_energies)

			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			counter += 1

		print('took {} iterations'.format(counter)) # \n final energy was {:6.6f}', E_old

	def gibbs(self, thresh=0.05, temp=4):

		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0

		while delta_E > thresh and counter < 10: # threshold for convergence
			random.shuffle(self.indices)
			for i, j in self.indices:
				local_energies = [math.exp(-1*self.local_energy(i,j,k)/temp) for k in range(self.no_classes)]
				sum_energies = sum(local_energies)
				r = random.uniform(0,1)
				z = 0
				for k in range(self.no_classes):
					z += local_energies[k] / sum_energies
					if z > r:
						self.labels[i,j] = k
						break
			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			counter += 1

		print('took {} iterations\n final energy was {:6.6f}'.format(counter,E_old))

# gmm with estimate_parameters function
# -> mle to get initial segmentation

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	K = 5
	# test_img = Image() # should raise error
	test_img = Image('./test_resized_2.jpg')
	means = [np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1) ]) for _ in range(K)]
	variances = [random.uniform(1,2) * np.eye(3)] * K
	
	# # #
	 ### this is how i interface w/ ur code
	# # #

	# test_gmm = oliver_gmm(test_img._data,3)

	# init_pi = [0.33, 0.33, 0.34]
	# init_mu = [np.array([0.5, 0.5, 0.5])]*3
	# init_sigma = [np.matrix([[1000.0, 0.0, 0.0], [0.0, 157.0, 0.0], [0.0, 0.0, 1.0]]) for _ in range(3)]

	# test_gmm.estimate_parameters(10) 


	test_mrf = MRF(test_img,means,variances)
	plt.imshow(test_mrf.labels,cmap='gist_gray_r')
	plt.savefig('before.png')
	test_mrf.icm()
	plt.imshow(test_mrf.labels)
	plt.savefig('after_icm.png',cmap='gist_gray_r')
	# lol, for my 640 x 425 px image this code took 140 seconds to run on my desktop