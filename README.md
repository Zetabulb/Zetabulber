# Zetabulber

This repository contains the Zetabulber - a tool used for visualizing Mandelbulbs associated with the Riemann zeta function (Zetabulbs). Essentially, this tool visualises a Mandelbulb that can travel through the complex surface of the Riemann zeta function.

## User guide

The Zetabulber is operated using keyboard and mouse. `W`, `A`, `S`, `D`, `Space`, `Shift` and `mouse` controls are used to operate the camera. The power of the Mandelbulb can be changed using the keys `+` and `-`. The keys `I`, `J`, `K` and `L` are used to navigate the Mandelbulb through the complex plane.

## System requirements

1. A CUDA capable GPU
2. Windows 10 or newer

## Source code building requirements

1. CUDA toolkit version 12.3. Newer versions can be used by changing project dependencies.

## Modified Borwein's algorithm

The Zetabulber uses a modification of Borwein's efficient algorithm for calculating the Riemann zeta function. This algorithm is very efficient when calculating multiple values of the Riemann zeta function at once. The implementation of this algorithm can be found in the file `zeta.cuh`

