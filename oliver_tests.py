from mrf_3d import Image, MRF, fisher2_MRF, SecondOrderMRF
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import numpy as np
from resegment import ShapeSegmentMRF
from random_shape_gen import MultiShapeHolder
K=4
test_img = Image('./scrot/0.png')
# plt.imshow(test_img._data)
# plt.show()
test_gmm = GaussianMixture(K,'diag',init_params='random',verbose=1)
d2_array = np.reshape(np.ravel(test_img._data),(test_img._data.shape[0]*test_img._data.shape[1],test_img._data.shape[2]))
res = test_gmm.fit(d2_array)
test_mrf = SecondOrderMRF(test_img, res.means_, [np.eye(3)*res.covariances_[i] for i in range(K)])
plt.imshow(test_mrf.labels)
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

def resegment(mask, imm, label, means, variances):
	# im_mask = mask
	# mask = np.zeros([mask.shape[0],mask.shape[1], 3])
	# mask[:, :, 0] = im_mask
	# mask[:, :, 1] = im_mask
	# mask[:, :, 2] = im_mask
	imm = Image(data=imm*255)
	im_mrf = ShapeSegmentMRF(imm, means, variances,label, mask)
	im_mrf.icm()
	plt.imshow(im_mrf.labels)
	plt.savefig('resegment.png',cmap='gist_gray_r')


truth_test = MultiShapeHolder(50,50)
truth_test.get_truth('./scrot/0.txt')
# for s in truth_test.shapes:
# 	print(s)
r = truth_test.get_shape(25,0)
print(r)
if r is not None:
	import matplotlib.pyplot as plt

	mask = r.get_mask(50,50)
	# print(mask)
	plt.imshow(mask)
	plt.savefig('mask.png')
resegment(mask, im, test_mrf.labels[35,5], test_mrf.means, test_mrf.variances)