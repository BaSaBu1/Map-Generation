import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from map import Map

# Parameters
N = 750
WIDTH = 1
WATER_LEVEL = 0.4
NOISE_SCALE = 3
CLUSTER = 5

# Plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.20) 

points = np.random.rand(N, 2) * WIDTH
map = Map(points, WIDTH, WATER_LEVEL, NOISE_SCALE, CLUSTER)

def plot_current_state():
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, WIDTH)
    ax.axis('off')
    
    map.plotVoronoi(ax)
    map.plotLand(ax)
    fig.canvas.draw_idle()

# Initial plot
plot_current_state()

# Sliders
ax_noise = plt.axes([0.25, 0.10, 0.65, 0.03])
slider_noise = Slider(ax_noise, 'Noise Scale', 1.0, 10.0, valinit=NOISE_SCALE, valstep=0.5)

def update(val):
    map.noise_scale = slider_noise.val
    map.assignAltitudes() 
    plot_current_state()

slider_noise.on_changed(update)

ax_button = plt.axes([0.8, 0.5, 0.1, 0.04])
btn = Button(ax_button, 'New Map')

def new_map(event):
    p = np.random.rand(N, 2) * WIDTH
    global map
    map = Map(p, WIDTH, WATER_LEVEL, slider_noise.val, CLUSTER)
    plot_current_state()

btn.on_clicked(new_map)

plt.show()