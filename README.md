# Mandelbrot set visualization using mpi4py
This repository contains a simple parallel implementation of a mandelbrot set visualizer. The program generates PNG images visualizing the mandelbrot fractal centered at customizable points and with customizable scale and resolution. The parallel implementation uses mpi4py (MPI + Python).

The repository includes the following codes:
- `seq_mandelbrot.py`, a reference sequential implementation of the program.
- `par_mandelbrot.py`, the parallelized version using mpi4py.

These programs were developed for educational purposes. They were intended to be used as example programs in an educational computing cluster comprising 8 Raspberry Pi 3B+ (up to 32 cores/processes), to demonstrate the speedup capabilities of parallel computing / supercomputers to high-school students.

The code for these programs is greatly based on [Draw the Mandelbrot Set in Python](https://realpython.com/mandelbrot-set-python/).

## Dependencies
- mpi4py
- matplotlib (for the color palette)
- numpy (included with matplotlib)
- pillow

## TODO
Currently, the customizable parameters of the programs are hard-coded. In the future, the user will be able to set them using command-line arguments.

The output of the programs will also be modified to provide more useful metrics and information, taking into account the context for which this program is developed.

