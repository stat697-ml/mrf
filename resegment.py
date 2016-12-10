import numpy as np
from mrf_3d import MRF
import math, random
class ShapeSegmentMRF(MRF):
	def __init__(self,image, means,variances,color_label, mask):
		super(ShapeSegmentMRF,self).__init__(image, means,variances)
		self.color_label = color_label
		self.mask=mask
	def local_energy(self,i,j, lab):
		beta = 10
		if lab == self.color_label:
			return -beta + beta*np.matrix(self.means[self.color_label]-self.image[i,j])*np.matrix(self.means[self.color_label]-self.image[i,j]).T
		return beta
	def icm(self, thresh=0.00000000000005):
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
				if self.mask[i,j]:
					local_energies = [self.local_energy(i,j,lab) for lab in range(self.no_classes)]
					self.labels[i,j] = np.argmin(local_energies)#self.estimate_label(i,j)

			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			self.update_params()
			print("THE COUNTER:", counter)
			print(delta_E)
			counter += 1

		print('took {} iterations'.format(counter)) # \n final energy was {:6.6f}', E_old


