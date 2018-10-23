#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:07:05 2018

@author: eert

State: y = [q1, q2, q1d, q2d]
"""

import numpy as np 
from numpy import sin, cos, pi
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

T = 10
nDisControls = 5
u_dis = [0, 3, 1, 2, 5]
t_dis = np.linspace(0, T, nDisControls)

def der_fun(t, y):
    #u = np.interp(t_dis, u_dis, t)
    
    q1 = y[0]
    q2 = y[1]
    q1d = y[2]
    q2d = y[3]
    
    delta = q2 - q1
    
    q1dd = ( m2 * l1 * q1d * q1d * sin(delta) * cos(delta) \
         + m2 *g * sin(q2) * cos(delta) \
         + m2 * l2 * q2d * q2d * sin(delta) \
         - (m1 + m2 ) * g *sin(q1) ) / ( (m1 + m2 ) * l1 \
         - m2 * l1 * cos(delta) * cos(delta) )
    q2dd = ( - m2 * l2 * q2d * q2d  * sin(delta) * cos(delta) \
         + ( m1 + m2 ) * (g * sin(q1) * cos(delta) \
         - l1 * q1d * q1d * sin(delta) \
         - g * sin(q2))) / ( (m1 + m2 ) * l2 \
         - m2 * l2 * cos(delta) * cos(delta) )
    
    return [q1d, q2d, q1dd, q2dd]

y0 = [pi/2, pi/2, 0, 0]

sol = integrate.solve_ivp(der_fun, [0, 10], y0)

# Positions
x1 = l1*sin(sol.y[0,:])
y1 = -l1*cos(sol.y[0,:])
x2 = l2*sin(sol.y[1,:]) + x1
y2 = -l2*cos(sol.y[1,:]) + y1

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(min(min(x1), min(x2)), max(max(x1), max(x2))), ylim=(min(min(y1), min(y2)), max(max(y1), max(y2))))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    return line,

anim = animation.FuncAnimation(fig, animate, frames = len(sol.y[0]),
                              interval=1000*np.diff(sol.t).mean(), blit=True, init_func=init)

plt.show()



