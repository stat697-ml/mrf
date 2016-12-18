import numpy as np
import random
from scipy import stats
from scipy import misc
import math
# import pdb

from image import Image
# sd = 1
# np.random.seed(sd)
# random.seed(sd)

class GaussianMixtureModel():
	def __init__(self, img, K,verbose=False):
		self.verbose = verbose
		self.data = img
		self.K = K
		self.pi = np.array([1.0/K] * K)
		self.mu = np.random.rand(K,3)
		self.sigma = [np.eye(3)*random.uniform(1,2)]*K

	def E_step(self, pi_old, mu_old, sigma_old):
		normals = [stats.multivariate_normal(mean=mu_old[i], cov=sigma_old[i]) for i in range(self.K)]
		gamma_z = np.zeros([self.data.height, self.data.width,self.K])
		for i in range(self.data.height):
			for j in range(self.data.width):
				for l in range(self.K):
					gamma_z[i, j, l] = pi_old[l] * normals[l].pdf(x=self.data[i, j])
				gamma_denom = gamma_z[i, j, :].sum(axis=0)
				gamma_z[i, j, :] = gamma_z[i, j, :] / gamma_denom
		return gamma_z

	def M_step(self, gamma_z):
		# estimate pi
		N = np.zeros(self.K)
		for i in range(self.data.height):  
			for j in range(self.data.width):
				for k in range(self.K):
					N[k] += gamma_z[i, j, k]
		pi_new = N / (self.data.height*self.data.width)

		# estimate mu
		mu_new = [np.array([0.0, 0, 0]) for _ in range(self.K)]
		for i in range(self.data.height):  
			for j in range(self.data.width):
				for k in range(self.K):
					mu_new[k] += gamma_z[i, j, k] * self.data[i, j] / N[k]

		# estimate sigma
		sigma_new = [[0.0,0,0] for _ in range(self.K)]
		for i in range(self.data.height):  
			for j in range(self.data.width):
				diffs = [np.matrix(self.data[i, j] - mu_new[k]) for k in range(self.K)]
				for k in range(self.K):
					sigma_new[k][0] += gamma_z[i, j, k] * (diffs[k].item(0) ** 2) / N[k]
					sigma_new[k][1] += gamma_z[i, j, k] * (diffs[k].item(1) ** 2) / N[k]
					sigma_new[k][2] += gamma_z[i, j, k] * (diffs[k].item(2) ** 2) / N[k]

		return [pi_new, mu_new, [sigma_new[k]*np.eye(3) for k in range(self.K)]]

	def likelihood(self,pi,mu,sigma):
		total = 0;
		normals = [stats.multivariate_normal(mean=mu[i], cov=sigma[i]) for i in range(self.K)]
		for i in range(self.data.height):
			for j in range(self.data.width):
				diffs = [np.matrix(self.data[i, j, 0:3] - mu[k]) for k in range(self.K)]
				log_arg = 0
				for k in range(self.K):
					log_arg+= pi[k]*normals[k].pdf(self.data[i,j])
				total += math.log(log_arg)
		return total

	def estimate_parameters(self, max_iter=200):
		i = 0
		thresh = 0.005
		pi_est = self.pi
		mu_est = self.mu
		sigma_est = self.sigma
		delta_ELBO = 999
		while i < max_iter and delta_ELBO > thresh:
			if self.verbose: print('counter',i)
			old_ELBO = self.likelihood(pi_est, mu_est, sigma_est)
			gamma_z = self.E_step(pi_est, mu_est, sigma_est)
			[pi_est, mu_est, sigma_est] = self.M_step(gamma_z)
			new_ELBO = self.likelihood(pi_est, mu_est, sigma_est)
			delta_ELBO = math.fabs(old_ELBO-new_ELBO)
			if self.verbose: print('delta ELBO',delta_ELBO)
			i += 1
		
		return [pi_est, mu_est, sigma_est]

if __name__ == '__main__':   
    #### Main Code ####
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from sklearn.mixture import GaussianMixture

    NN = 30
    K = 2
    # m = [0,0]
    # sigma = [0,0]
    # m[0] = (np.array([4,4]))
    # m[1] = (np.array([0,0]))
    # sigma[0] = np.matrix([[1,0],[0,1]])
    # sigma[1] = np.matrix([[0.5,0.25],[0.25,1.5]])

    # m = np.array([0.0, 0.0, 1.0])
    # mu = [m for i in range(K)]  # row vectors...
    # sigma = [np.matrix([[12.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]), np.matrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) ]
    # pi = [0.75, 0.25]
    # z = np.random.choice(size=[NN, NN], a=[0, 1], p=pi)
    # im = np.zeros([NN, NN, 3])
    # for ii in range(NN):
    # 	for jj in range(NN):
    # 		im[ii, jj] = np.random.multivariate_normal(mean=mu[z[ii, jj]], cov=sigma[z[ii, jj]])

    # init_pi = [0.5, 0.5]
    # init_mu = [np.array([10.0, 10.0, 1.0]), np.array([0.0, 0.0, 0.0])]
    # init_sigma = [np.matrix([[1000.0, 0.0, 0.0], [0.0, 157.0, 0.0], [0.0, 0.0, 1.0]]) for i in range(K)]

    # GMM.init_labels_params()
    im = mpimg.imread("./yosemite.png")
    im1 = Image(data=im,scale=5,pepper=False)

    GMM = GaussianMixtureModel(im1, K)
    ests = GMM.estimate_parameters(200)
    # print(ests)
    # for est in ests:
    print(ests[2])
    # plt.figure(1)
    # plt.imshow(sigma[0])
    # plt.figure(2)
    # plt.imshow(ests[2][0])
    # plt.show()
    test_gmm = GaussianMixture(K, 'diag', init_params='random', verbose=1)
    d2_array = np.reshape(np.ravel(im1._data),(im1._data.shape[0] * im1._data.shape[1], im1._data.shape[2]))
    res = test_gmm.fit(d2_array)
    print(res.covariances_)

	###
	#Next, write ELBO in, so can measure convergence.
	#
	#