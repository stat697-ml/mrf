import pyglet
from pyglet import gl
from PIL import Image

import random, math

from random_shape_gen import RandomShapeGenerator

class CustomGroup(pyglet.graphics.Group):
	def set_state(self):
		gl.glEnable(gl.GL_TEXTURE_2D)

	def unset_state(self):
		gl.glDisable(gl.GL_TEXTURE_2D)

class Drawer(pyglet.window.Window):
	def __init__(self, x=500,y=500,visible=False):
		super(Drawer, self).__init__(x,y)
		self.set_visible(visible)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		self.main_batch = pyglet.graphics.Batch()
		self.scrot_name = None

		
	def on_draw(self):
		self.update_drawing()

	def update_drawing(self):
		# self.draw_rect(0,0,self.width,self.height,(0,0,0))
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		self.clear()
		self.main_batch.draw()
		print('drwa')
		self.main_batch = pyglet.graphics.Batch()
		self.save_drawing()

	def update_save_value(self,filename):
		self.scrot_name = filename

	def save_drawing(self): # must be .PNG
		if self.scrot_name is not None:
			pyglet.image.get_buffer_manager().get_color_buffer().save(self.scrot_name)
			self.scrot_name = None
			# print('save')

	def draw_rect(self,bottom_x,bottom_y,width,height,color=None):
		x,y,w,h = bottom_x,bottom_y,width,height
		if color is None: color = (255,255,255)
		this_shape = CustomGroup()
		self.main_batch.add(4, gl.GL_QUADS, this_shape,
			('v2i',[x,y,x+w,y,x+w,y+h,x,y+h]),
			('c3B',color*4))

	def draw_square(self,bottom_x,bottom_y,width,color=None):
		self.draw_rect(bottom_x,bottom_y,width,width,color)

	def my_range(self,start,stop,step):
		res = start
		while res <= stop:
			yield res
			res += step

	def draw_ellipse(self,x1,y1,x2,y2,color=None):
		if color is None: color = (255,255,255)
		xrad = abs((x2-x1) / 2.0)
		yrad = abs((y2-y1) / 2.0)
		x = (x1+x2) / 2.0
		y = (y1+y2) / 2.0

		step = 32.0
		rad = max((xrad+yrad)/2, 0.01)
		rad_ = max(min(step / rad / 2.0, 1), -1)
		da = min(2 * math.asin(rad_), math.pi / 16)

		points = [(x + math.cos(a) * xrad, y + math.sin(a) * yrad) for a in self.my_range(0,2*math.pi,da)]
		points = list(y for x in points for y in x)

		this_shape = CustomGroup()

		self.main_batch.add(len(points)//2,gl.GL_TRIANGLE_FAN, this_shape, 
			('v2f',points), 
			('c3B',(color*(len(points)//2)))
			)

	def draw_circle(self,center_x,center_y,radius,color=None):
		self.draw_ellipse(center_x-radius,center_y-radius,center_x+radius,center_y+radius,color)

if __name__ == '__main__': 
	import os
	if not os.path.exists('./scrot'):
	    os.makedirs('./scrot')
	    
	window = Drawer(500, 500,True)
	rsg = RandomShapeGenerator(500,500)
	
	# window.save_drawing('ttt.png')
	# window.draw_rect(100,300,200,100,(255,0,0))
	# window.draw_square(350,75,100,(128,128,0))
	# window.draw_ellipse(250,40,69,150,(0,255,0))
	# window.draw_circle(400,344,60,(0,0,255))
	# pyglet.image.get_buffer_manager().get_color_buffer().save('./test.png')
	# @window.event
	# def on_draw():
		# pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
		#     [0, 1, 2, 0, 2, 3],
		#     ('v2i', (100, 100,
		#              150, 100,
		#              150, 150,
		#              100, 150))
		# )
		# 
	
	# @window.event
	# def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
	#     pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x-dx, y, x-dx, y-dy, x, y-dy]))
	# pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')
	counter = 0
	def test(val):
		global counter
		if counter % 2 == 0:
			window.update_save_value('./scrot/{}.png'.format(counter//2))
			shapes_to_draw = rsg.generate_random_shapes(random.randint(3,8))
			truth = 'K={}\n'.format(len(shapes_to_draw))
			for s in shapes_to_draw:
				truth = truth + s.save_truth()
				wh = s.width_height
				x,y,w,h = s.left, s.bot, wh[0], wh[1]
				if s.shape_type == 'Ellipse':
					window.draw_ellipse(x,y,s.right,s.top,s.color)
				else:
					window.draw_rect(x,y,w,h,s.color)
			with open('./scrot/{}.txt'.format(counter//2),'w') as text_dump:
				text_dump.write(truth)
		else:
			window.clear()
		counter += 1
		if counter > 100:
			pyglet.app.exit()

	pyglet.clock.schedule_interval(test, 1/50.0)
	pyglet.app.run()