#https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/denoise.pdf

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from numpy.random import randint, randn, rand

Size = 50 
J = 1
H = 0.0
Temp = 0

def spin_direction(field, x, y):
    energy = H
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        # boundary conditions
        if x + dx < 0: dx += Size
        if y + dy < 0: dy += Size
        if x + dx >= Size: dx -= Size
        if y + dy >= Size: dy -= Size
        energy += J * field.ix[x + dx, y + dy]

    if Temp == 0:
        p = (np.sign(energy) + 1) * 0.5
    else:
        p = 1/(1+np.exp(-2*(1/Temp)*energy))
    if rand() <= p:
        spin = 1
    else:
        spin = -1
    return spin

def run_gibbs_sampling(field, iternum=5):
    for _ in range(iternum):
        lattice = df([(y,x) for x in range(Size) for y in range(Size)])
        lattice.reindex(np.random.permutation(lattice.index))
        for x, y in lattice.values:
            field.ix[x, y] = spin_direction(field, x, y)

if __name__ == '__main__':
    fig = plt.figure()
    field = df(randint(2,size=(Size,Size))*2-1)

    temps = [0.,.5,1.,1.5,2.,2.5,5.0,10.0][::-1]
    for i in range(1,9):
        Temp = temps[i-1]
        run_gibbs_sampling(field)
        ax = fig.add_subplot(2,4,i)
        ax.set_title("T = %2.1f\nH = %1.1f" % (Temp, H))
        axim = ax.imshow(field.values, vmin=-1, vmax=1,
                         cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig('test.png') # strangely this has to be inside of the for-loop
                                # otherwise final image is just the last pic repeated