import numpy as np
from math import log
from PIL import Image
from matplotlib import colormaps
from time import time

CONVERGENCE_STEPS = 512
CONVERGENCE_THRESHOLD = 1000

def julia_sequence(parameter, candidate=0):
    while True:
        yield candidate
        candidate = candidate**2 + parameter

mandelbrot_sequence = lambda z: julia_sequence(z)

def stable(candidate, iters=50):
    if type(candidate) == complex:
        z = 0
    elif type(candidate) == np.ndarray:
        z = np.zeros_like(candidate)
    seq = julia_sequence(candidate, z)
    for _ in range(iters):
        res = next(seq)
    return abs(res) <= 2

def stability(candidate, iters=CONVERGENCE_STEPS, smooth=True):
    z = 0
    seq = julia_sequence(candidate, z)
    for iter in range(iters):
        res = next(seq)
        if abs(res) > CONVERGENCE_THRESHOLD:
            if smooth:
                return (iter + 1 - log(log(abs(res))) / log(2)) / iters
            return iter / iters
    return 1
        

# 2D array of complex numbers equally spaced in the specified rectangle
def complex_matrix(elements, center=(0, 0), scale=4):
    xcenter, ycenter = center
    yrang = xrang = scale / 2
    re = np.linspace(xcenter - xrang, xcenter + xrang, elements)
    im = np.linspace(ycenter - yrang, ycenter + yrang, elements)
    return re[:, np.newaxis] + im[np.newaxis, :] * 1j

name = "ocean"
palette = [tuple(int(round(channel * 255)) for channel in color) for color in [colormaps[name](c) for c in np.linspace(0, 1, CONVERGENCE_STEPS)]][::-1]

width = height = 256

image = Image.new(mode="RGB", size=(width, height))

t1 = time()
#mandelbrot_matrix = complex_matrix(width)
mandelbrot_matrix = complex_matrix(width, (-0.7435, 0.1314), 0.002)
t2 = time()
print(f"Generation of values to check: {t2 - t1}s")

for y in range(height):
    for x in range(width):
        image.putpixel((x, y), palette[int((len(palette) - 1) * stability(mandelbrot_matrix[x, -((y + 1) % height)]))])
t3 = time()
print(f"Mandelbrot stability calculation and image creation: {t3 - t2}s")

image.show()
image.save(f"{name}{width}.png")

