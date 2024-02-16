# Zetabulber

This repository contains the Zetabulber - a tool used for visualizing Mandelbulbs associated with the Riemann zeta function (Zetabulbs). 

## System requirements

1. A CUDA capable GPU
2. Windows 10 or newer

## Source code building requirements

1. CUDA toolkit version 12.3. Newer versions can be used by changing project dependencies.

## Modified Borwein's algorithm

The Zetabulber uses a modification of Borwein's efficient algorithm for calculating the Riemann zeta function. This algorithm is very efficient when calculating multiple values of the Riemann zeta function at once. The implementation of this algorithm can be found in the file `zeta.cuh`

This repository contains an implementation of this 