from mrf_3d import MRF, SecondOrderMRF
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import numpy as np
from resegment import ShapeSegmentMRF
from shapes import ShapeCollection
from image import Image
import shape_fitting
K=6
test_img = Image('./watershed.png',scale=5, pepper=False)
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
#
# def get_true_means(mrf_result):
# 	image = np.zeros([mrf_result.labels.shape[0],mrf_result.labels.shape[1],3])
# 	for i in range(image.shape[0]):
# 		for j in range(image.shape[1]):
# 			image[i,j] = mrf_result.means[mrf_result.labels[i,j]]
# 	return image
#
# def resegment(mask, imm, label, means, variances):
# 	# im_mask = mask
# 	# mask = np.zeros([mask.shape[0],mask.shape[1], 3])
# 	# mask[:, :, 0] = im_mask
# 	# mask[:, :, 1] = im_mask
# 	# mask[:, :, 2] = im_mask
# 	imm = Image(data=imm,scale=1,pepper=False)
# 	# print(imm.width*imm.height)
# 	# print(mask.shape)
# 	im_mrf = ShapeSegmentMRF(imm, means, variances,label, mask,test_mrf.labels)
# 	im_mrf.icm()
# 	image = np.zeros([im_mrf.labels.shape[0], im_mrf.labels.shape[1], 3])
# 	for i in range(image.shape[0]):
# 		for j in range(image.shape[1]):
# 			image[i, j] = im_mrf.means[im_mrf.labels[i, j]]
# 	return image
# #
# all_shapes=shape_fitting.get_all_shapes(test_mrf.labels)
#
# for shape in all_shapes:#[all_shapes[i] for i in range(len(all_shapes)) if i!=8]:
# # shape = all_shapes[6]
# 	mask = shape.get_mask(62,100)
# 	im = resegment(mask,im,shape.label,test_mrf.means,test_mrf.variances)
# 	# plt.imshow(mask)
# 	# plt.show()
# 	# plt.imshow(im)
# 	# plt.show()
# plt.imshow(im)
# plt.savefig('total_resegment.png')
#
# # truth_test = MultiShapeHolder(50,50)
# # truth_test.get_truth('./draw/scrot/0.txt')
# # for s in truth_test.shapes:
# # 	print(s)
# # r = truth_test.get_shape(12,41)
# # print(r)
# # if r is not None:
# # 	import matplotlib.pyplot as plt
# #
# # 	mask = r.get_mask(50,50)
# # 	# print(mask)
# # 	plt.imshow(mask)
# # 	plt.savefig('mask.png')
# # resegment(mask, im, test_mrf.labels[37,41], test_mrf.means, test_mrf.variances)