print("Loading libraries...")

import numpy as np
from math import log
from PIL import Image
from matplotlib import colormaps
from time import time, sleep
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
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
    xrang = scale / 2
    yrang = scale / nprocs
    yoffset = -scale / 2 + (nprocs - rank - 1) * yrang
    re = np.linspace(xcenter - xrang, xcenter + xrang, elements)
    im = np.linspace(ycenter + yoffset, ycenter + yoffset + yrang, elements // nprocs)
    return re[:, np.newaxis] + im[np.newaxis, :] * 1j

# Color palette for the image
cm_name = "ocean"
palette = [tuple(int(round(channel * 255)) for channel in color) for color in [colormaps[cm_name](c) for c in np.linspace(0, 1, CONVERGENCE_STEPS)]][::-1]

# Image width and height
width = height = 480

comm.barrier() # Process synchronization point


# 1. GENERATION OF THE VALUES TO COMPUTE

t1 = time()
mandelbrot_matrix = complex_matrix(width, center=(-0.7435, 0.1314), scale=0.002)
t2 = time()
# Gather times:
time_generation = t2 - t1
all_times_generation = None
if rank == 0:
    all_times_generation = np.empty((nprocs,))
time_gen_req = comm.Igather(np.array([time_generation]), all_times_generation)


# 2. COMPUTATION OF THE VALUES

# Generate image shape:
pixels = np.empty((*mandelbrot_matrix.transpose().shape, 4), dtype=np.uint8)

# Init computation loop variables:
last_progress = 0
global_progress = None
if rank == 0:
    global_progress = np.zeros((1,), dtype=np.uint8)
    last_global_progress = 0
    step_bar = False
    print("Progress:   |" + "·" * 100 + "|   0%", end="")
    sys.stdout.flush()
sync_point = None
# Computation loop:
for y in range(height // nprocs):
    for x in range(width):
        pixels[y, x] = palette[int((len(palette) - 1) * stability(mandelbrot_matrix[x, -((y + 1) % (height // nprocs))]))]

    # Update global progress
    if int(y / height * 100) > last_progress:
        last_progress = int(y / height * 100)
        if sync_point != None:
            sync_point.Wait()   # Wait for last reduction to complete
            if rank == 0:
                # Update variables with results of last reduction
                last_global_progress = global_progress[0]
                step_bar = True
        # Begin reduction of progress (asynchronously)
        sync_point = comm.Ireduce(np.array([last_progress], dtype=np.uint8), global_progress)

    # Print/step progress bar if needed
    if rank == 0 and step_bar:
        step_bar = False
        prog = last_progress * nprocs # Cheat to avoid race conditions(?)
        spaces = 2 if prog < 10 else 1
        print("\rProgress:   |" + "█" * prog + "·" * (100 - prog) + f"| {' ' * spaces}{prog}%", end="")
        sys.stdout.flush()
if rank == 0:
    print("\rProgress:   |" + "█" * 100 + "| 100%")
    sys.stdout.flush()
t3 = time()

# Gather times:
time_computation = t3 - t2
all_times_computation = None
if rank == 0:
    all_times_computation = np.empty((nprocs,))
time_comp_req = comm.Igather(np.array([time_computation]), all_times_computation)


# 4. ASSEMBLE AND SHOW FINAL IMAGE

# Gather pixels:
all_pixels = None
if rank == 0:
    all_pixels = np.empty((width, height, 4), dtype=np.uint8)
comm.Gather(pixels, all_pixels)

# Terminate all non-root processes
if rank != 0:
    exit()

image = Image.fromarray(all_pixels)

t_end = time()

sleep(1) # For dramatic effect
print(f"\nTOTAL EXECUTION TIME: {t_end - t_ini}s\n")

# Print execution time statistics:
time_gen_req.wait()
time_comp_req.wait()

print(f"Generation step:\n\tavg:\t{np.mean(all_times_generation)}s\n\tstdev:\t{np.std(all_times_generation)}s")
print(f"Computation step:\n\tavg:\t{np.mean(all_times_computation)}s\n\tstdev:\t{np.std(all_times_computation)}s\n\tmin:\t{np.min(all_times_computation)}s (process {np.argmin(all_times_computation)})\n\tmax:\t{np.max(all_times_computation)}s (process {np.argmax(all_times_computation)})")

image.show()
#image.save(f"{cm_name}{width}-{nprocs}.png")
