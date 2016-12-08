import numpy as np
import math, random
import pdb
from scipy import misc
from skimage import io, img_as_float, color
from skimage.exposure import equalize_hist
from skimage.util import random_noise

from gmm import GaussianMixtureModel as oliver_gmm

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
		self._data = self._data[:,:,0:3]
		self._data = misc.imresize(self._data,0.1)/255

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

	def beta(self, i, j, coeff=10):
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
		# print(i)
		# print(self.image[i,j])
		# print(self.means[label])
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

		return singletons + doubletons#/2

	def local_energy(self, i, j, label):
		return self.singleton(i,j,label) + self.doubleton(i,j,label)

	### estimation algos for updating labels
	# efficient graph cut


	def update_params(self):

		lab_points = [[] for _ in range(self.no_classes)]
		for i in range(self.image.height):
			for j in range(self.image.width):
				for k in range(self.no_classes):
					if self.labels[i,j] == k:
						lab_points[k].append(self.image[i,j])

		for k in range(self.no_classes):
			lab_points[k] = np.array(lab_points[k])

		# for k in range(self.no_classes):
		# 	if len(lab_points[k])==0:
		# 		del lab_points[k]
		# 		self.no_classes -= 1

		means = [(1/len(lab_points[k]))*lab_points[k].sum(axis=0) if len(lab_points[k])>0 else self.means[k] for k in range(self.no_classes)]
		# diffs = [[np.matrix(lab_points[k][i]-means[k]) for i in range(len(lab_points[k]))] for k in range(self.no_classes)  ]
		# variances = [np.matrix([[0.0,0,0],[0,0,0],[0,0,0]]) for _ in range(self.no_classes) ]
		# for k in range(self.no_classes):
		# 	if len(lab_points[k])>0:
		# 		for i in range(len(lab_points[k])):
		# 			variances[k]+=diffs[k][i]*diffs[k][i].T/len(lab_points[k])
		# print(self.means[1])
		self.means = means#, self.variances = means, variances




	def icm(self, thresh=0.0000005):
		# basically loop through everything picking minimizing labeling until "convergence"
		print("icm")
		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0
		while delta_E > thresh and counter < 10: # threshold for convergence
			delta_E = 0
			# mix up the order the indices are visited
			random.shuffle(self.indices)

			for i, j in self.indices:
				local_energies = [self.local_energy(i,j,k) for k in range(self.no_classes)]
				self.labels[i,j] = np.argmin(local_energies)#self.estimate_label(i,j)

			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			self.update_params()
			print("THE COUNTER:", counter)
			print(delta_E)
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
			print("THE COUNTER:", counter)
			counter += 1

		print('took {} iterations\n final energy was {:6.6f}'.format(counter,E_old))


class fisher_MRF(MRF):
	def doubleton(self, i, j, label):
		energy = 0.0
		alpha = 1
		beta = 1
		for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
			if all([i + dy < self.image.height,
					i + dy >= 0,
					j + dx < self.image.width,
					j + dx >= 0]):
				if label == self.labels[i+dy,j+dx]:
					energy -= beta
				else:
					sig = np.matrix(self.variances[label] + self.variances[self.labels[i + dy, j + dx]])
					fisher=(self.means[label] - self.means[self.labels[i+dy,j+dx]])*np.matrix(self.means[label] - self.means[self.labels[i+dy,j+dx]]).T#*np.linalg.inv(sig)*np.matrix(self.means[label] - self.means[self.labels[i+dy,j+dx]]).T
					# print(fisher)
					energy += alpha*fisher
		return energy

class hard_boundary_MRF(MRF):
	def doubleton(self, i, j, label):
		energy = 0.0
		alpha = 100
		beta = 1
		for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
			if any([i + dy >= self.image.height,
					i + dy < 0,
					j + dx >= self.image.width,
					j + dx < 0]):
				return 0
		count=0
		negcount=0
		# for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
		# 	if label == self.labels[i+dy,j+dx]:
		# 		count+=1
			# else:
			# 	energy+=beta

		# if count==4:
		# 	energy-=count*beta
		# else:
		# 	energy+=count*beta

		countH = 0
		if label == self.labels[i-1,j] and label== self.labels[i+1,j]:
			energy += alpha
		if label == self.labels[i,j-1] and label==self.labels[i,j+1]:
			energy += alpha
		else:
			energy-=alpha
		return energy


class fisher2_MRF(MRF):
	def doubleton(self, i, j, label):
		energy = 0.0
		alpha = 10
		beta = 1
		for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
			if all([i + dy < self.image.height,
					i + dy >= 0,
					j + dx < self.image.width,
					j + dx >= 0]):
				# sig = np.matrix(self.variances[label] + self.variances[self.labels[i + dy, j + dx]])
				# fisher = (self.means[label] - self.means[self.labels[i + dy, j + dx]]) * np.matrix(self.means[label] - self.means[self.labels[i + dy, j + dx]]).T  # *np.linalg.inv(sig)*np.matrix(self.means[label] - self.means[self.labels[i+dy,j+dx]]).T
				# print(fisher)
				if label == self.labels[i+dy,j+dx]:
					energy += -beta+alpha*(self.image[i,j]-self.image[i+dy,j+dx])*np.matrix(self.image[i,j]-self.image[i+dy,j+dx]).T
				else:

					energy += beta
		return energy

class pixel_MRF(MRF):
	def doubleton(self, i, j, label):
		energy = 0.0
		thresh = 0.02
		for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
			if all([i + dy < self.image.height,
					i + dy >= 0,
					j + dx < self.image.width,
					j + dx >= 0]):
				if label == self.labels[i+dy,j+dx] and np.linalg.norm(self.image[i,j]-self.image[i+dy, j+dx])**2 < thresh:
					energy -= self.beta(i,j)
				else:
					energy += self.beta(i,j)

		return energy

class SecondOrderMRF(MRF):
	def __init__(self,*args):
		super(SecondOrderMRF,self).__init__(*args)
		self.check_arrays = [
		[[1,0,0],
		 [0,1,0],
		 [0,0,1]],
		[[0,1,0],
		 [0,1,0],
		 [0,1,0]],
		[[0,0,1],
		 [0,1,0],
		 [1,0,0]],
		[[0,0,0],
		 [1,1,1],
		 [0,0,0]],
		 [[1,1,1],
		  [1,1,1],
		  [1,1,1]]
		]

	def doubleton(self, i, j, label):
		i_start, i_end = max(0,i-1), min(i+1,self.image.height)
		j_start, j_end = max(0,j-1), min(j+1,self.image.width)

		masked_label_array = self.labels[i_start:i_end+1,j_start:j_end+1] == label
		masked_label_array[1,1] = True
		# this will be True if the labels all match one of the check arrays
		check = max([np.all(masked_label_array==ca) for ca in self.check_arrays])
		if check:
			return -1
		return 1

#
#
# class flaherty_MRF(MRF):
# 	def doubleton(self, i, j):
# 		energy = 0.0
# 		for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
# 			if all([i + dy < self.image.height,
# 					i + dy >= 0,
# 					j + dx < self.image.width,
# 					j + dx >= 0]):
# 				if label == self.labels[i + dy, j + dx]:
# 					energy -= self.beta(i, j)
# 				else:
# 					energy += self.beta(i, j)#NOT FINISHED
# 		return energy





# gmm with estimate_parameters function
# -> mle to get initial segmentation

if __name__ == '__main__':
	import matplotlib.pyplot as plt
<<<<<<< HEAD
	K = 1
=======
	K = 6
>>>>>>> 3a92203d78331d7f16a5a9e1f90b0ae7c7a71e88
	# test_img = Image() # should raise error
	test_img = Image('./watershed.png')#Image('./test_resized_2.jpg')

	# means = [np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1) ]) for _ in range(K)]
	# means = [np.array([1/k,1/k,1/k]) for k in range(1,K+1)]
	variances = [ np.eye(3)] * K#random.uniform(1,2) *

	# # #
	 ### this is how i interface w/ ur code
	# # #

	test_gmm = oliver_gmm(test_img,K)
	# plt.imshow(test_img._data)
	# plt.show()
	# init_pi = [0.33, 0.33, 0.34]
	# init_mu = [np.array([0.5, 0.5, 0.5])]*3
	# init_sigma = [np.matrix([[1000.0, 0.0, 0.0], [0.0, 157.0, 0.0], [0.0, 0.0, 1.0]]) for _ in range(3)]

	[pi_est, means, not_variances] = test_gmm.estimate_parameters(25)
	test_mrf = SecondOrderMRF(test_img,means,variances)

	# test_mrf = fisher2_MRF(test_img,means,variances)
	# test_mrf.doubleton = new_doubleton
	plt.imshow(test_mrf.labels)
	# plt.show()
	plt.savefig('before.png',cmap='gist_gray_r')
	test_mrf.icm()
	# test_mrf.gibbs()
	plt.imshow(test_mrf.labels)
	plt.savefig('after_icm.png',cmap='gist_gray_r')
	im = np.zeros([test_img.height,test_img.width,3])
	for i in range(test_img.height):
		for j in range(test_img.width):
			im[i,j] = test_mrf.means[test_mrf.labels[i,j]]
	plt.imshow(im)
	plt.savefig('real_means.png',cmap='gist_gray_r')
	# plt.savefig('after_gibbs.png',cmap='gist_gray_r')
	# lol, for my 640 x 425 px image this code took 140 seconds to run on my desktop

#scikit learn code
# from sklearn.mixture import GaussianMixture
# test_img = io.imread('./test_resized_2.jpg')
# # change full to 'diag' for diagonal covariance
# test_gmm = GaussianMixture(3,'full',init_params='random',verbose=1)
# d2_array = np.reshape(np.ravel(test_img),(test_img.shape[0]*test_img.shape[1],test_img.shape[2]))
# res = test_gmm.fit(d2_array)
# print(res.means_)
# print(res.covariances_)