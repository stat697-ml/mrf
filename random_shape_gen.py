import random
import math
import numpy as np

class MultiShapeHolder():
	def __init__(self,total_height,total_width=None):
		self.shapes = []
		self.k = 0
		self.total_height, self.total_width = total_width, total_height

	def get_truth(self,filename):
		with open(filename,'r') as f:
			truth = f.read()
			truth = truth.split('\n')[:-1]
			get_k = truth[0]
			truth = truth[1:]
			self.k = int(get_k[get_k.index('=')+1:])
			for t in truth:
				shape_vals = t.split(',')
				st = shape_vals[0]
				ltrb = [int(v) for v in shape_vals[1:5]]
				c = eval(','.join(shape_vals[-3:]))
				new_shape = Shape(*ltrb,shape_type=st,color=c)
				self.shapes.append(new_shape)
#### note: you may need to flip i,j by subtracting from total_height/width
	def get_shape(self,i,j):
		# gives you whatever shape
		# first flip between my shape coord system and how images are handled lol
		i = self.total_height - i
		for s in self.shapes:
			if s.is_within(i,j): return s
		return None
	

class Shape():
	def __init__(self,left,top,right,bot,shape_type=None,color=None):
		self.left, self.top, self.right, self.bot = left, top, right, bot
		self.shape_type = shape_type
		if color is None:
			color = (255,255,255)
		self.color = color
	@property
	def bottom(self):
		return self.bot

	@property
	def center(self):
		return [(self.left + self.right) // 2, (self.top + self.bot) // 2]

	@property
	def width_height(self):
		return [abs(self.right - self.left), abs(self.top - self.bot)]

	def is_within(self,i,j):
		# returns whether a pixel is inside of the shape
		return (self.bot <= i <= self.top) and (self.left <= j <= self.right)

	def get_mask(self,total_height,total_width):
		to_return = np.zeros((total_height, total_width))
		if self.shape_type in ['Rectangle','Square']:
			to_return[self.bot:self.top,self.left:self.right] = 1
		elif self.shape_type in ['Ellipse','Circle']:
			x, y = np.meshgrid(np.arange(total_width), np.arange(total_height))
			c, wh = self.center, self.width_height
			x -= c[0]
			y -= c[1]
			to_return = ((x * x)/(wh[0]**2)*4 + (y * y)/(wh[1]**2)*4 < 1)
		return np.flipud(to_return)

	def __str__(self):
		if self.shape_type is None:
			prefix = 'NullShape'
		else:
			prefix = self.shape_type
		return '{} centered at {} (x,y),\n\t width/height {}\n\t color {}'.format(prefix,self.center,self.width_height,self.color)

	def save_truth(self):
		return '{},{},{},{},{},{}\n'.format(self.shape_type,self.left,self.top,self.right,self.bot,self.color)

class RandomShapeGenerator():
	def __init__(self,width,height):
		self.total_width, self.total_height = width, height
		

	def generate_random_shapes(self,k):
		self.k = k
		# first partition grid 
		self.grid_pops = [[0] * 3  for _ in range(math.ceil(k/3))]
		self.grid_width, self.grid_height = len(self.grid_pops[0]), len(self.grid_pops)
		grid_indices = [i for i in range(self.grid_width * self.grid_height)]

		# now randomly fill it in
		filled = random.sample(grid_indices,self.k)
		for f in filled:
			self.grid_pops[f//self.grid_width][f%self.grid_width] = 1

		# assign shapes randomly
		shape_choices = ['Ellipse','Rectangle'] #circle, square omitted for now
		grid_shapes = []

		for i in grid_indices:
			r, c = i // self.grid_width, i % self.grid_width
			left = c * self.total_width // self.grid_width
			bot = r * self.total_height // self.grid_height
			right = left + self.total_width // self.grid_width
			top = bot + self.total_height // self.grid_height
			random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
			new_shape = Shape(left,top,right,bot,random.choice(shape_choices),random_color)
			grid_shapes.append(new_shape)

		# now for each cell that has a shape, will randomly determine 
		# if it will be tangent to each of its edges
		choices = [True, False]

		for c in range(self.grid_width):
			for r in range(self.grid_height):
				# if not populated pass
				if self.grid_pops[r][c] == 0:
					pass
				this_shape = grid_shapes[r*self.grid_width + c]
				# check right neighbor
				if c < self.grid_width - 1:
					if self.grid_pops[r][c+1] == 1:
						# randomly set left edge (of neighbor) to be tangent
						if random.choice(choices):
							grid_shapes[r*self.grid_width + c + 1].left = this_shape.right
						else:
							this_shape.right = random.randint(this_shape.center[0]+1,this_shape.right-1)
				# check bottom neighbor
				if r < self.grid_height - 1:
					if self.grid_pops[r+1][c] == 1:
						# randomly set bottom edge (of neighbor) to be tangent
						if random.choice(choices):
							grid_shapes[(r+1)*self.grid_width + c].bot = this_shape.top
						else:
							this_shape.top = random.randint(this_shape.center[1]+1,this_shape.top-1)


		return [s for i,s in enumerate(grid_shapes) if self.grid_pops[i//self.grid_width][i%self.grid_width]]

if __name__ == '__main__':
	

	# test_rsg = RandomShapeGenerator(500,500)
	# grid_shapes = test_rsg.generate_random_shapes(4)


	# for i,s in enumerate(grid_shapes):
	# 	if test_rsg.grid_pops[i//3][i%3]:
	# 		print(s)

	truth_test = MultiShapeHolder(500,500)
	truth_test.get_truth('./scrot/0.txt')
	# for s in truth_test.shapes:
	# 	print(s)
	r = truth_test.get_shape(50,50)
	# r = truth_test.shapes[0]
	print(r)
	if r is not None:
		import matplotlib.pyplot as plt

		mask = r.get_mask(500,500)
		plt.imshow(mask)
		plt.savefig('mask.png')