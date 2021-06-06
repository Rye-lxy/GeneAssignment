import numpy as np
from math import log
from main import read_fa, read_genome, encoder, encoder_int
np.seterr(divide='ignore',invalid='ignore')
import time
from numba import cuda
import math

# generate transition matrix. arr is reference genome, n is n-th
def transition_matrix(arr:str, n:int):

    M = np.zeros(shape=(4**n, 4))
    for i in range(len(arr)-n):
        M[(int(arr[i:i+n],4)),int(arr[i+n])] += 1

    M = M/ np.sum(M,axis=1,keepdims=True)
    M[np.isnan(M)] = 0
    return M

# restore all transition matrixs
def matrix_pole(X,n:int):
    Matrix = np.zeros(shape=(len(X),4**n, 4)) 
    for i in range(len(X)):
        print(">>> Establishing matrix",X[i][1].split("|")[-1])
        codeX = encoder(X[i][0])
        Matrix[i] = transition_matrix(codeX,n)
        print("Success")
        
    return Matrix

#Assign a target sequence to a specific genome
def genome_assign(M, sequence: str, n:int):
    pro = 1
    seq = encoder(sequence)
    for i in range(len(seq)-n):
        pro *= M[(int(seq[i:i+n],4)),int(seq[i+n])] # the bigger, the better
    S = -log(pro) # the smaller s is, the better 
    return S

# Compare reads to multiple matrix (n:m)
def compare_read_to_matrix(Matrix, reads,n:int):
    maxscore = []
    for read in reads:
        score = []
        for M in Matrix:
            score.append(genome_assign(M, read, n))
        maxscore.append(score.index(min(score))) # find the most similar reference genome
    return maxscore


@cuda.jit
def kmm_gpu(coded_seq, rowindex_list, colindex_list, k, n):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(idx,n,stride):
        rowindex = 0
        for j in range(0,k):
            rowindex += coded_seq[i+j] * (4 ** (k-j-1))
        rowindex_list[i] = rowindex
        colindex_list[i] = coded_seq[i+k]

@cuda.jit
def get_mat_gpu(rowindex_list,colindex_list,M_,n,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < 64:  ##except for the last group
        for i in range(0, n):
            M_[idx][rowindex_list[idx*n+i],colindex_list[idx*n+i]] += 1
    else:
        for i in range(idx * n, N):
            M_[idx][rowindex_list[i],colindex_list[i]] += 1


@cuda.jit
def maxscore_gpu(reads_int,genmat_list,maxscore,k,n):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n :
        max_score = 0
        max_pos = 0
        pos = 0
        for M in genmat_list :
            pro = 1
            pro_rev = 1
            len = 0
            for x in reads_int[idx]:
                len+=1
            for i in range(0,len-k):
                row_index = 0
                col_index = reads_int[idx][i+k]
                for j in range(0, k):
                    row_index += reads_int[idx][i + j] * (4 ** (k - j - 1))
                pro *= M[row_index,col_index]
            for i in range(len-k-1,-1,-1):
                row_index = 0
                col_index = 3 - reads_int[idx][i-k]
                for j in range(0, k):
                    base = 3 - reads_int[idx][i-j]
                    row_index += base * (4 ** (k - j - 1))
                pro_rev *= M[row_index,col_index]
            if pro_rev > pro :
                pro = pro_rev
            if pro > max_score:
                max_score = pro
                max_pos = pos
            pos += 1

        maxscore[idx] = max_pos



if __name__ == '__main__':
    """   
    ## processed by cpu
    t1 = time.time()
    reads = read_fa("test.fa")
    X = read_genome("./genomes/")
    print("read_genome finish")
    M = matrix_pole(X,3)
    t2 = time.time()
    print("time1",str(t2-t1))
    score = compare_read_to_matrix(M, reads,3)
    t3 = time.time()
    print("time2",str(t3-t2))
    print(score)
    """
    
    ##processed by gpu
    t1 = time.time()
    reads = read_fa("test.fa")
    X = read_genome("./genomes/")
    ##get matrix
    genmat_list = []
    for genome in X :
        coded_seq = encoder_int(genome[0])
        seq_len = len(genome[0])
        Gcoded_seq = cuda.to_device(coded_seq)
        k = 9
        rowindex_list = cuda.device_array(seq_len - k, dtype=int)
        colindex_list = cuda.device_array(seq_len - k, dtype=int)
        TPB = 128
        BPG = 128
        # run gpu function
        cuda.synchronize()
        kmm_gpu[TPB, BPG](Gcoded_seq, rowindex_list, colindex_list, k, seq_len - k)
        cuda.synchronize()
        rowindex_list = (rowindex_list.copy_to_host())
        colindex_list = (colindex_list.copy_to_host())
        # generate kmm matrix
        M_ = np.zeros(shape=(64,4**k,4),dtype = int)
        GM_ = cuda.to_device(M_)
        indexperthread = int(math.ceil((seq_len - k) / 64))
        cuda.synchronize()
        get_mat_gpu[8, 8](rowindex_list, colindex_list,GM_, indexperthread, seq_len - k)
        cuda.synchronize()
        M_ = GM_.copy_to_host()
        M = np.zeros((4 ** k, 4), dtype=float)
        for i in range(0,64):
            M += M_[i]
        M = M / np.sum(M, axis=1, keepdims=True)
        M[np.isnan(M)] = 0
        genmat_list.append(M)
    t2 = time.time()
    print("time1", str(t2 - t1))

    ##compare reads to genomes
    reads_int = []
    for read in reads:
        reads_int.append(encoder_int(read))
    Greads_int = cuda.to_device(reads_int)
    maxscore = [0] * len(reads_int)
    Gmaxscore = cuda.to_device(maxscore)
    Ggenmat_list = cuda.to_device(genmat_list)
    tpb = 256
    bpg = int(math.ceil(len(reads_int)/tpb))
    cuda.synchronize()
    maxscore_gpu[tpb,bpg](Greads_int, Ggenmat_list, Gmaxscore, 9, len(reads_int))
    cuda.synchronize()
    maxscore = Gmaxscore.copy_to_host().tolist()
    t3 = time.time()
    print("time2 ",str(t3-t2))
    print(maxscore)






