import numpy as np
from math import log
from main import read_fa, read_genome, encoder
import time

np.seterr(divide='ignore', invalid='ignore')


# generate transition matrix. arr is reference genome, n is n-th
def transition_matrix(arr: str, n: int):
    M = np.zeros(shape=(4 ** n, 4))
    for i in range(len(arr) - n):
        M[(int(arr[i:i + n], 4)), int(arr[i + n])] += 1

    M = M / np.sum(M, axis=1, keepdims=True)
    M[np.isnan(M)] = 0
    return M


# restore all transition matrixs
def matrix_pole(X, n: int):
    Matrix = np.zeros(shape=(len(X), 4 ** n, 4))
    for i in range(len(X)):
        print(">>> Establishing matrix", X[i][1].split("|")[-1])
        codeX = encoder(X[i][0])
        Matrix[i] = transition_matrix(codeX, n)
        print("Success")

    return Matrix


# Assign a target sequence to a specific genome
def genome_assign(M, sequence: str, n: int):
    pro = 1
    seq = encoder(sequence)
    for i in range(len(seq) - n):
        pro *= M[(int(seq[i:i + n], 4)), int(seq[i + n])]  # the bigger, the better
    S = -log(pro)  # the smaller s is, the better
    return S


# Compare reads to multiple matrix (n:m)
def compare_read_to_matrix(Matrix, reads, n: int):
    maxscore = []
    for read in reads:
        score = []
        for M in Matrix:
            score.append(genome_assign(M, read, n))
        maxscore.append(score.index(min(score)))  # find the most similar reference genome
    return maxscore


if __name__ == '__main__':
    t1 = time.time()
    reads = read_fa("test.fa")
    X = read_genome("./genomes/")
    print("read_genome finish")
    M = matrix_pole(X, 3)
    t2 = time.time()
    print("time1", str(t2 - t1))
    score = compare_read_to_matrix(M, reads, 3)
    t3 = time.time()
    print("time2", str(t3 - t2))
    print(score)
