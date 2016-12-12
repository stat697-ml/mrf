from scipy import misc
from skimage import io, img_as_float#, color
# from skimage.exposure import equalize_hist
from skimage.util import random_noise

class Image():
	"""
	holds a picture
	can either supply a filename or
	data as a numpy array
	scale param = how many times to scale image down (default of 10 will make image that is 10 times smaller)
	"""
	def __init__(self, filename=None, data=None,pepper=False,scale=10):
		assert any([filename is not None, data is not None]), "you need to supply an image file or pass a picture array"

		if filename is not None:
			self._data = io.imread(filename)
		else:
			self._data = data

		# preprocessing
		# self._data = color.rgb2lab(self._data)
		# if self._data.ndim > 2:
		# 	self._data = equalize_hist(color.rgb2gray(self._data)) # convert to grayscale
		
		self._data = self._data[:,:,0:3]
		self._data = img_as_float(self._data) 

		if pepper:
			self._data = random_noise(self._data) # pepper

		if scale > 1:
			self._data = misc.imresize(self._data,1.0/scale)

		(self.height, self.width, self.bitdepth) = self._data.shape

		self.indices = [(i,j) for i in range(self.height) for j in range(self.width)]


	def __getitem__(self, item):
	# piggyback off of numpy's array indexing
		return self._data.__getitem__(item)