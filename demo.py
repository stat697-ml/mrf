import pyglet
from pyglet import gl
from shapes import Shape, ShapeCollection, RandomShapeGenerator
from scripts import MRFScripts
import math
import os
import random
import threading
import time

class CustomGroup(pyglet.graphics.Group):
	def set_state(self):
		gl.glEnable(gl.GL_TEXTURE_2D)

	def unset_state(self):
		gl.glDisable(gl.GL_TEXTURE_2D)

## to-do
# add preprocessing code
# figure out if want to show matplotlib


# class FuncThread(threading.Thread):
# 	def __init__(self, target, fname, *args):
# 		self._target = target
# 		self._fname = fname
# 		self._args = args
# 		threading.Thread.__init__(self)
 
# 	def run(self):
# 		while not os.path.isfile(self._fname):
# 			print('wait..')
# 			time.sleep(1)
# 		self._target(self._fname,*self._args)

class Button():
	def __init__(self,xx,yy,text,action=None):
		self.x, self.y = xx,yy
		self.text = text
		self.label =  pyglet.text.Label(text,
                          font_name='Consolas',
                          font_size=16,
                          x=xx, y=yy,
                          anchor_x='center', anchor_y='center')
		self.action = action
	def draw(self):
		l,r = self.x - 75, self.x + 75
		b,t = self.y + 50, self.y - 50
		pyglet.graphics.draw(4, gl.GL_QUADS,
		    ('v2i', [l,b,r,b,r,t,l,t]),
			('c3B',(128,128,128)*4))
		self.label.draw()

	def check_click(self,x,y):
		if self.x - 75 < x < self.x + 75:
			if self.y + 50 > y > self.y - 50:
				if self.action is not None:
					self.action()

class Drawer(pyglet.window.Window):
	def __init__(self, x=500,y=500,visible=False):
		super(Drawer, self).__init__(x,y)
		self.scripter = MRFScripts()

		self.width, self.height = x, y
		self.canvas_area_bounds = [180, self.width - 180] # lr

		self.shape_col = ShapeCollection(self.height,self.width)

		self.set_visible(visible)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

		self.main_batch = pyglet.graphics.Batch()
		self.scrot_name = None

		self.rect_imgs = [pyglet.resource.image('demo_resources/rect_select.png'),
						  pyglet.resource.image('demo_resources/rect_selected.png')
						 ]
		self.oval_imgs = [pyglet.resource.image('demo_resources/oval_select.png'),
						  pyglet.resource.image('demo_resources/oval_selected.png')
						 ]

		self.shape_selection = 0

		self.color_pallette = [(230,184,195),(154,192,222),(168,211,142),
							   (114,184,172),(142,101,157),(146,127,104)]

		self.color_selection = 0

		self.drag_begin = None
		self.current_drag = None
		self.buttons = [Button(self.width-90,self.height-60,'clear',self.delet_screen),
						Button(self.width-90,self.height-175,'randomize',self.draw_random),
						Button(self.width-90,self.height-290,'just gmm',self.run_gmm),
						Button(self.width-90,self.height-405,'vanilla mrf',self.run_vmrf),
						Button(self.width-90,self.height-520,'boundaries',self.run_hbmrf),
						Button(self.width-90,self.height-635,'shape priors',self.run_spmrf),

						]
						




	def on_mouse_press(self,x, y, button, modifiers):
		if button == 1:
			# check if click within any of our left-side buttons
			if 30 < x < self.canvas_area_bounds[0]:
				if 5*self.height//6 + 10 < y < self.height - 10:
					self.shape_selection = 0
				elif 4*self.height//6 + 10 < y < 5*self.height//6- 10:
					self.shape_selection = 1
				elif 15 < y < 6*self.height//10:
					self.color_selection = (y - 15) // (self.height//10)
			# drawing
			elif self.canvas_area_bounds[0] < x < self.canvas_area_bounds[1]:
				self.drag_begin = (x,y)
			elif self.canvas_area_bounds[1] < x:
				for b in self.buttons:
					b.check_click(x,y)
		elif button == 4:
			ss = self.shape_col.get_shape(self.height-y,x-self.canvas_area_bounds[0])
			if ss is not None:
				self.shape_col.shapes.remove(ss[-1])
				

	def on_mouse_release(self, x, y, button, modifiers):
		# print(x,y,button,'RELEASE!')
		if self.canvas_area_bounds[0] < x < self.canvas_area_bounds[1]:
			if self.drag_begin is not None:
				if all([abs(x-self.drag_begin[0])>40,abs(y-self.drag_begin[1])>40]):
					self.draw_drag(x,y)

		self.drag_begin = None
		self.current_drag = None

	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
	 	if self.drag_begin is not None:
	 		if self.canvas_area_bounds[0] < x < self.canvas_area_bounds[1]:
		 		self.current_drag = (x,y)
		 	else:
		 		self.current_drag = None
		
	def draw_drag(self,final_x,final_y):
		start_x, start_y = self.drag_begin[0], self.drag_begin[1]
		color = self.color_pallette[self.color_selection]
		l,t = min(start_x,final_x) - self.canvas_area_bounds[0], self.height - min(start_y,final_y)
		r,b = max(start_x,final_x) - self.canvas_area_bounds[0], self.height - max(start_y,final_y)
		shape_types = ['Rectangle','Ellipse']
		new_shape = Shape(l,t,r,b,shape_types[self.shape_selection],color=self.color_pallette[self.color_selection])
		# print(new_shape)
		self.shape_col.shapes.append(new_shape)
		self.color_selection = (self.color_selection + 1)% len(self.color_pallette)

	def on_draw(self):
		self.update_drawing()
		self.draw_gui()

	def update_drawing(self):
		# self.draw_rect(0,0,self.width,self.height,(0,0,0))
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		self.clear()

		for s in self.shape_col.shapes:
			if s.shape_type == 'Rectangle':
				width, height = abs(s.left - s.right), abs(s.top - s.bot)
				x, y = s.left + self.canvas_area_bounds[0], self.height - s.top
				self.draw_rect(x,y,width,height,s.color)
			elif s.shape_type == 'Ellipse':
				x1, x2 = s.left + self.canvas_area_bounds[0], s.right + self.canvas_area_bounds[0]
				y1, y2 = self.height - s.top, self.height - s.bot
				self.draw_ellipse(x1,y1,x2,y2,s.color)

		# self.main_batch.draw()
		# self.main_batch = pyglet.graphics.Batch()
		self.save_drawing()

	def draw_gui(self):
		self.rect_imgs[1-self.shape_selection].blit(30,5*self.height//6)
		self.oval_imgs[self.shape_selection].blit(30,4*self.height//6)
		pyglet.graphics.draw(2, gl.GL_LINES,
		    ('v2i', (self.canvas_area_bounds[0], 0, self.canvas_area_bounds[0], self.height))
		)
		pyglet.graphics.draw(2, gl.GL_LINES,
		    ('v2i', (self.width - self.canvas_area_bounds[0], 0, self.width - self.canvas_area_bounds[0], self.height))
		)
		for i in range(len(self.color_pallette)):
			l,b,r,t = 30,15+i*self.height//10,150,(i+1)*self.height//10
			if i == self.color_selection:
				l -= 5
				r += 5
				b -= 5
				t += 5
			pyglet.graphics.draw(4, gl.GL_QUADS,
			    ('v2i', [l,b,r,b,r,t,l,t]),
				('c3B',self.color_pallette[i]*4))
		if self.drag_begin is not None and self.current_drag is not None:
			pyglet.graphics.draw(2, gl.GL_LINES,
			    ('v2i', (self.drag_begin[0], self.drag_begin[1], self.current_drag[0], self.current_drag[1])),
			    ('c3B',self.color_pallette[self.color_selection]*2)
			)

		for button in self.buttons:
			button.draw()
			
	def delet_screen(self):
		self.shape_col.shapes = []

	def save_drawing(self): # must be .PNG
		if self.scrot_name is not None:
			imbuff = pyglet.image.get_buffer_manager().get_color_buffer()
			imbuff.save(self.scrot_name)
			print('saved',self.scrot_name)
			self.scrot_name = None

	def draw_rect(self,bottom_x,bottom_y,width,height,color=None):
		x,y,w,h = bottom_x,bottom_y,width,height
		if color is None: color = (255,255,255)
		# this_shape = CustomGroup()
		# self.main_batch.add(4, gl.GL_QUADS, this_shape,
		pyglet.graphics.draw(4, gl.GL_QUADS,
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

		# this_shape = CustomGroup()

		# self.main_batch.add(len(points)//2,gl.GL_TRIANGLE_FAN, this_shape, 
		pyglet.graphics.draw(len(points)//2,gl.GL_TRIANGLE_FAN,
			('v2f',points), 
			('c3B',(color*(len(points)//2)))
			)

	def draw_circle(self,center_x,center_y,radius,color=None):
		self.draw_ellipse(center_x-radius,center_y-radius,center_x+radius,center_y+radius,color)

	def draw_random(self):
		rsg = RandomShapeGenerator(self.width-2*self.canvas_area_bounds[0], self.height)
		self.shape_col.shapes = rsg.generate_random_shapes(random.randint(3,8))

	def save_shot(self):
		### note:
		# to test you will have to first cut off the left and right parts of the image
		# but the truth has proper coords
		# then on scale, will need to scale all values down from truth (if using it :v))
		if not os.path.exists('./demo_scrot'):
		    os.makedirs('./demo_scrot')
		counter = 0
		fname = './demo_scrot/{}.png'.format(counter)
		while os.path.isfile(fname):
			counter += 1
			fname = './demo_scrot/{}.png'.format(counter)
		self.scrot_name = fname
		truth = 'K={}\n'.format(len(self.shape_col.shapes))
		for s in self.shape_col.shapes:
			truth = truth + s.save_truth()
		with open('./demo_scrot/{}.txt'.format(counter),'w') as text_dump:
			text_dump.write(truth)
		return fname

	def run_gmm(self):
		fname = self.save_shot()
		print(fname)
		def just_gmm():
			while not os.path.isfile(fname):
				print('wait..')
				time.sleep(1)
			self.scripter.gmm(fname,len(self.shape_col.shapes)+1)

		t1 = threading.Thread(target=just_gmm)
		t1.start()
		# t1 = FuncThread(self.scripter.gmm, fname,3)
		# t1.start()
		# t1.join()

	def run_vmrf(self):
		fname = self.save_shot()
		print(fname)
		def just_gmm():
			while not os.path.isfile(fname):
				print('wait..')
				time.sleep(1)
			self.scripter.vanilla_MRF(fname,len(self.shape_col.shapes)+1)

		t1 = threading.Thread(target=just_gmm)
		t1.start()

	def run_hbmrf(self):
		fname = self.save_shot()
		print(fname)
		def just_gmm():
			while not os.path.isfile(fname):
				print('wait..')
				time.sleep(1)
			self.scripter.hard_boundary_MRF(fname,len(self.shape_col.shapes)+1)

		t1 = threading.Thread(target=just_gmm)
		t1.start()

	def run_spmrf(self):
		fname = self.save_shot()
		print(fname)
		def just_gmm():
			while not os.path.isfile(fname):
				print('wait..')
				time.sleep(1)
			self.scripter.segment_with_priors(fname,len(self.shape_col.shapes)+1)

		t1 = threading.Thread(target=just_gmm)
		t1.start()


if __name__ == '__main__': 
	window = Drawer(1280, 720, True)
	pyglet.app.run()
