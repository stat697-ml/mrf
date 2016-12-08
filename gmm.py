import numpy as np
from random import randint
import random
from scipy import stats
#from mrf_3d import Image
from scipy import misc
import math
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from numba import jit
sd = 1
np.random.seed(sd)
random.seed(sd)

class GaussianMixtureModel(object):
	def __init__(self, img, K):#, pi_init, mu_init, sigma_init):
		self.data = img
		self.K = K
		self.pi = np.array([1.0/K] * K)
		self.mu = np.random.rand(K,3)
		self.sigma = [np.eye(3)*random.uniform(1,2)]*K

	# def init_labels_params(self):
	# self.labels = np.zeros(self.data.shape)
	# pdb.set_trace()
	# for i in range(self.labels.size):
	# 	self.labels[i] = randint(0, self.K - 1);
	# self.pi = np.tile(0.0, self.K)
	# self.mu = [np.array([0.0, 0, 0]) for i in range(self.K)]  # row vectors...
	# self.sigma = [np.matrix([[1, 0.0, 0], [0, 1.0, 0], [0, 0.0, 1]]) for i in range(K)]
	# @profile
	def E_step(self, pi_old, mu_old, sigma_old):
		# pdb.set_trace()
		normals = [stats.multivariate_normal(mean=mu_old[i], cov=sigma_old[i]) for i in range(self.K)]
		gamma_z = np.zeros([self.data.height, self.data.width,self.K])
		for i in range(self.data.height):
			for j in range(self.data.width):
				# gamma_denom = 0.0
				# for k in range(self.K):
					# pdb.set_trace()
					# gamma_denom += pi_old[k] * normals[k].pdf(x=self.data[i, j])  # ignore alpha channel
				for l in range(self.K):
					gamma_z[i, j, l] = pi_old[l] * normals[l].pdf(x=self.data[i, j]) # / gamma_denom
				gamma_denom = gamma_z[i, j, :].sum(axis=0)
				gamma_z[i, j, :] = gamma_z[i, j, :] / gamma_denom
		return gamma_z

	def M_step(self, gamma_z):
		# estimate pi
		N = np.zeros(self.K)
		for i in range(self.data.height):  # TODO:vectorize
			for j in range(self.data.width):
				for k in range(self.K):
					N[k] += gamma_z[i, j, k]
		# print(gamma_z)
		pi_new = N / (self.data.height*self.data.width)

		# estimate mu
		mu_new = [np.array([0.0, 0, 0]) for _ in range(self.K)]
		for i in range(self.data.height):  # TODO:vectorize
			for j in range(self.data.width):
				for k in range(self.K):
					# pdb.set_trace()
					mu_new[k] += gamma_z[i, j, k] * self.data[i, j] / N[k]

		sigma_new = np.array([[0,0,0] for _ in range(self.K)])
		for i in range(self.data.height):  # TODO:vectorize
			for j in range(self.data.width):
				diffs = [np.matrix(self.data[i, j] - mu_new[k]) for k in range(self.K)]
				for k in range(self.K):
					sigma_new[k][0] += gamma_z[i, j, k] * (diffs[k][0] ** 2) / N[k]
					sigma_new[k][1] += gamma_z[i, j, k] * (diffs[k][1] ** 2) / N[k]
					sigma_new[k][2] += gamma_z[i, j, k] * (diffs[k][2] ** 2) / N[k]

		return [pi_new, mu_new, [sigma_new[k]*np.eye(3) for k in range(self.K)]]

	# def likelihood(self,gamma_z,pi,mu,sigma):
	# 	total = 0;
	# 	for i in range(self.data.height):#TODO:vectorize
	# 		for j in range(self.data.width):
	# 			diffs = [np.matrix(self.data[i, j, 0:3] - mu[k]) for k in range(self.K)]
	# 			log_arg = 0
	# 			for k in range(self.K):
	# 				log_arg+= pi[k](math.log(pi[k]) - math.log
	#@profile
	def estimate_parameters(self, max_iter):
		i = 0
		pi_est = self.pi
		mu_est = self.mu
		sigma_est = self.sigma
		while i < max_iter:
			print(i)
			# print(pi_est)
			print(mu_est)
			print(sigma_est)
			gamma_z = self.E_step(pi_est, mu_est, sigma_est)
			# pdb.set_trace()
			[pi_est, mu_est, sigma_est] = self.M_step(gamma_z)
			i += 1
		### PLEASE DON'T LEAVE THIS IN###
		temp = np.copy(sigma_est[0][0])
		sigma_est[0][0] = np.copy(sigma_est[0][1])
		sigma_est[0][1] = temp
		return [pi_est, mu_est, sigma_est]

if __name__ == '__main__':   
    #### Main Code ####
    NN = 30
    K = 4
    # m = [0,0]
    # sigma = [0,0]
    # m[0] = (np.array([4,4]))
    # m[1] = (np.array([0,0]))
    # sigma[0] = np.matrix([[1,0],[0,1]])
    # sigma[1] = np.matrix([[0.5,0.25],[0.25,1.5]])

    # m = np.array([0.0, 0.0, 1.0])
    # mu = [m for i in range(K)]  # row vectors...
    # sigma = [np.matrix([[2.0, 10.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]), np.matrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) ]
    # pi = [0.75, 0.25]
    # z = np.random.choice(size=[100, 100], a=[0, 1], p=pi)
    # im = np.zeros([NN, NN, 3])
    # for ii in range(NN):
    # 	for jj in range(NN):
    # 		# plt.hold(true)
    # 		im[ii, jj] = np.random.multivariate_normal(mean=mu[z[ii, jj]], cov=sigma[z[ii, jj]])

    # init_pi = [0.5, 0.5]
    # init_mu = [np.array([10.0, 10.0, 1.0]), np.array([0.0, 0.0, 0.0])]
    # init_sigma = [np.matrix([[1000.0, 0.0, 0.0], [0.0, 157.0, 0.0], [0.0, 0.0, 1.0]]) for i in range(K)]

    # GMM.init_labels_params()
    im = misc.imresize(mpimg.imread("./watershed.png"),5)/255
    im = Image(data=im)

    GMM = GaussianMixtureModel(im, K)
    ests = GMM.estimate_parameters(100)
    # print(ests)
    # for est in ests:
    print(ests[2])
    # plt.figure(1)
    # plt.imshow(sigma[0])
    # plt.figure(2)
    # plt.imshow(ests[2][0])
    # plt.show()
