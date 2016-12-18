import numpy as np
from mrf_3d import MRF
import math, random
from image import Image
import pdb

def resegment(test_mrf, mask, imm, label, means, variances):
	# im_mask = mask
	# mask = np.zeros([mask.shape[0],mask.shape[1], 3])
	# mask[:, :, 0] = im_mask
	# mask[:, :, 1] = im_mask
	# mask[:, :, 2] = im_mask
	imm = Image(data=imm,scale=1,pepper=False)
	# print(imm.width*imm.height)
	# print(mask.shape)
	im_mrf = ShapeSegmentMRF(imm, means, variances,label, mask,test_mrf.labels)
	im_mrf.icm()
	image = np.zeros([im_mrf.labels.shape[0], im_mrf.labels.shape[1], 3])
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			image[i, j] = test_mrf.means[im_mrf.labels[i, j]]
	return image

class ShapeSegmentMRF(MRF):
	def __init__(self,image, means,variances,color_label, mask,old_labels):
		super(ShapeSegmentMRF,self).__init__(image, means,variances)
		self.color_label = color_label
		self.mask=mask
		self.labels = old_labels
		self.stddev = 0.0
		for m in self.means:
			for n in self.means:
				self.stddev+=np.matrix(n-m)*np.matrix(n-m).T
		self.stddev = math.sqrt(self.stddev/(len(self.means)**2))
	def local_energy(self,i,j,lab):
		if lab == self.color_label:
			# pdb.set_trace()
			# print(np.matrix(self.means[self.color_label]-self.image[i,j])*np.matrix(self.means[self.color_label]-self.image[i,j]).T)
			# print('mean',self.means[self.color_label],'intensity',self.image[i,j])
			# print("stddev: ",self.stddev)
			return -1 + (np.matrix(self.means[self.color_label]-self.image[i,j])*np.matrix(self.means[self.color_label]-self.image[i,j]).T)
		return 1
	def icm(self, thresh=0.00000000000005):
		# basically loop through everything picking minimizing labeling until "convergence"
		# print("icm")
		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0
		while counter < 1: # threshold for convergence,....delta_E > thresh and
			delta_E = 0
			# mix up the order the indices are visited
			random.shuffle(self.indices)
			# print(self.mask.shape)
			for i, j in self.indices:
				if self.mask[i,j]:
					local_energies = [self.local_energy(i,j,lab) for lab in range(self.no_classes)]
					self.labels[i,j] = np.argmin(local_energies)#self.estimate_label(i,j)

			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			self.update_params()
			# print("THE COUNTER:", counter)
			# print(delta_E)
			counter += 1

		print('took {} iterations'.format(counter)) # \n final energy was {:6.6f}', E_old


