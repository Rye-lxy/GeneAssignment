# GeneAssignment
Project of **BI371** in Shanghai Jiao Tong University, a short read assigning method without comparison. Users can change the file paths in main function in `kmm.py` and `kmer.py`. And we will add some command line parameter and optimize the API of our program. 

## Contents

- [Dependent Packages](#Dependent-Packages)
- [K-mers Model](#K-mers-Model)
- [KMM](#KMM)
- [GPU](#GPU)

## Dependent Packages
* `numpy`
* `Bio`
* `numba`

## K-mers Model
Usage: `python kmer.py`.

## KMM
Usage: `python kmm.py`.

## GPU
Make sure your NVIDIA GPU support CUDA in advance.
Set up the environment locally:
```
conda install numba
conda install cudatoolkit
```
CUDA Toolkit can also be downloaded from the website(https://developer.nvidia.com/cuda-toolkit)

 

