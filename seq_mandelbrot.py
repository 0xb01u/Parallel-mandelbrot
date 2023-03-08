print("Loading libraries...")
import numpy as np
from math import log
from PIL import Image
from matplotlib import colormaps
from time import time, sleep
import sys

t_ini = time()

'''
Max number of sequence elements computed to test convergence of a candidate.
HIGHER: More precise, slower.
'''
CONVERGENCE_STEPS = 512
'''
Threshold to consider any candidate divergent.
Whenever a sequence reaches or surpasses this value, its associated candidate
is considered divergent.
'''
CONVERGENCE_THRESHOLD = 1000

'''
Computes the next value of the Julia sequence.
When the candidate is 0, the Julia sequence is the Mandelbrot sequence for
the parameter.
'''
def julia_sequence(parameter, candidate=0):
    while True:
        yield candidate
        candidate = candidate**2 + parameter

mandelbrot_sequence = lambda z: julia_sequence(z) # Lambda shortcut for Mandelbrot

'''
Tests whether a value of the sequence is stable.
'''
def stable(candidate, iters=CONVERGENCE_STEPS):
    if type(candidate) == complex:
        z = 0
    elif type(candidate) == np.ndarray:
        z = np.zeros_like(candidate)
    seq = julia_sequence(candidate, z)
    for _ in range(iters):
        res = next(seq)
    return abs(res) <= 2

'''
Returns the stability of a value in the sequence.
The stability equals to the number of iterations (candidates computed)
until divergence is reached.
'''
def stability(candidate, iters=CONVERGENCE_STEPS, smooth=True):
    z = 0
    seq = julia_sequence(candidate, z)
    for iter in range(iters):
        res = next(seq)
        if abs(res) > CONVERGENCE_THRESHOLD:
            if smooth: # Smooth out the borders
                return (iter + 1 - log(log(abs(res))) / log(2)) / iters
            return iter / iters
    return 1
        
'''
Generates a 2D array of complex numbers equally spaced in the specified
rectangle.
'''
def complex_matrix(elements, center=(0, 0), scale=4):
    xcenter, ycenter = center
    yrang = xrang = scale / 2
    re = np.linspace(xcenter - xrang, xcenter + xrang, elements)
    im = np.linspace(ycenter - yrang, ycenter + yrang, elements)
    return re[:, np.newaxis] + im[np.newaxis, :] * 1j

# Color palette for the image
cm_name = "ocean"
palette = [tuple(int(round(channel * 255)) for channel in color) for color in [colormaps[cm_name](c) for c in np.linspace(0, 1, CONVERGENCE_STEPS)]][::-1]

# Image width and height
width = height = 480

print("Libraries loaded\n\nGenerating image...")


# 1. GENERATION OF THE VALUES TO COMPUTE

t1 = time()
mandelbrot_matrix = complex_matrix(width, center=(-0.7435, 0.1314), scale=0.002)
t2 = time()
time_generation = t2 - t1


# 2. COMPUTATION OF THE VALUES

# Generate image shape:
pixels = np.empty((*mandelbrot_matrix.transpose().shape, 4), dtype=np.uint8)

# Init computation loop variables:
last_progress = 0
print("Progress:   |" + "·" * 100 + "|   0%", end="")
# Computation loop:
for y in range(height):
    for x in range(width):
        pixels[y, x] = palette[int((len(palette) - 1) * stability(mandelbrot_matrix[x, -((y + 1) % (height))]))]

    # Update progress
    if int(y / height * 100) > last_progress:
        last_progress = int(y / height * 100)
        prog = last_progress
        spaces = 2 if prog < 10 else 1
        print("\rProgress:   |" + "█" * prog + "·" * (100 - prog) + f"| {' ' * spaces}{prog}%", end="")
        sys.stdout.flush()
print("\rProgress:   |" + "█" * 100 + "| 100%")
sys.stdout.flush()
t3 = time()
time_computation = t3 - t2


# 4. ASSEMBLE AND SHOW FINAL IMAGE

image = Image.fromarray(pixels)

t_end = time()

sleep(1) # For dramatic effect
print(f"\nTOTAL EXECUTION TIME: {t_end - t_ini}s\n")

# Print execution time statistics:
print(f"Generation step:\n\ttime:\t{time_generation}s")
print(f"Computation step:\n\ttime:\t{time_computation}s")

image.show()
#image.save(f"{cm_name}{width}.png")

