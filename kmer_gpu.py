import numpy as np
import os
from math import sqrt
from main import read_fa, read_fna, read_genome, encoder, encoder_int
import time
from numba import cuda
import math


# k: the length of k-mer.
# sequence: a genome sequence with A, T, G, C.
def kmer(k: int, sequence: str):
    coded_sequence = encoder(sequence)

    seq_len = len(coded_sequence)  # Get length of sequence.

    lst = np.zeros(4 ** k, dtype=float)  # Generate a list of all the k-mers.

    # Calculate the frequency of each k-mer
    for i in range(0, seq_len - k + 1):
        index = int(coded_sequence[i:i+k], 4)
        lst[index] += 1

    # # Get the corresponding order of the index with descending order
    # sorted_index = np.flip(np.argsort(lst))

    # Calculate the probability
    lst = lst / (seq_len - k + 1)

    return lst.tolist()


# Get the k-mer distribution distance in probability space
def get_dist(list_1: list, list_2: list):
    if len(list_1) == len(list_2):
        length = len(list_1)
        sqsum = 0  # sum of square

        # Calculate the sum of square in probability space
        for i in range(0, length):
            sq = (list_1[i] - list_2[i]) ** 2
            sqsum = sqsum + sq

        # Return the distance (or we can also return sum of square directly)
        return sqrt(sqsum)
    else:
        raise ValueError('The length of 2 k-mer list are different!')


# Assign a target sequence to a specific genome
def genome_assign(gen_vec_list: list, sequence: str):
    dist_list = []
    seq_vec = kmer(3, sequence)
    for gen_vec in gen_vec_list:
        dist_list.append(get_dist(seq_vec, gen_vec))

    min_pos = dist_list.index(min(dist_list))

    # TODO: Determine a threshold of minimum distance

    # Return the description of the genome
    return min_pos


###GPU functions
@cuda.jit
def min_dist(gen_veclist,reads_veclist,minpos_list,n):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n :
        seq_vec = reads_veclist[idx]
        len = 4 ** 3
        minpos = 0
        minsum = 10000
        pos = 0
        for gen_vec in gen_veclist :
            sqsum = 0
            for i in range(0,len):
                sq = (seq_vec[i] - gen_vec[i]) ** 2
                sqsum = sqsum + sq
            if sqsum < minsum :
                minsum = sqsum
                minpos = pos
            pos += 1

        minpos_list[idx] = minpos


@cuda.jit
def kmer_gpu(coded_seq, index_list, k, n):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(idx,n,stride):
        index = 0
        for j in range(0,k):
            index += coded_seq[i+j] * (4 ** (k-j-1))
        index_list[i] = index






if __name__ == '__main__':
    t1 = time.time()
    reads = read_fa("test.fa")
    t2 = time.time()
    print("read_fa_time: "+str(t2-t1))

    t1 = time.time()
    path = "./genomes/"
    gen_list = read_genome(path)
    t2 = time.time()
    print("read_genome_time: " + str(t2 - t1))

    ##calculate kmer of genomes by cpu
    """
    t1 = time.time()
    gen_vec_list = []
    for genome in gen_list:
        gen_vec = kmer(3, genome[0])
        gen_vec_list.append(gen_vec)
    t2 = time.time()
    print("genome_kmer_cpu_time: " + str(t2 - t1))
    """

    ##calculate kmer of genomes by gpu
    t1 = time.time()
    gen_vec_list = []
    for genome in gen_list:
        coded_seq = encoder_int(genome[0])
        seq_len = len(genome[0])
        Gcoded_seq = cuda.to_device(coded_seq)
        k = 3
        index_list = cuda.device_array(seq_len-k+1,dtype=int)
        TPB = 128
        #BPG = int(math.ceil(seq_len/TPB))
        BPG = 128
        #run gpu function
        cuda.synchronize()
        kmer_gpu[TPB,BPG](Gcoded_seq, index_list, k, seq_len-k+1)
        cuda.synchronize()
        index_list = (index_list.copy_to_host())
        #generate kmer vector
        vec = np.zeros(4 ** k, dtype=float)
        for id in index_list:
            vec[id] += 1
        vec = (vec / (seq_len - k + 1)).tolist()
        gen_vec_list.append(vec)

    t2 = time.time()
    print("genome_kmer_gpu_time" + str(t2 - t1))


    ###classification processed by cpu
    start_time_cpu = time.time()
    min_pos_list = []
    for seq in reads:
        min_pos = genome_assign(gen_vec_list, seq)
        min_pos_list.append(min_pos)
    print(min_pos_list)
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu
    print("CPU process time: "+ str(cpu_time))

    ###classification processed by gpu
    #start_time_gpu = time.time()
    #copy essential data to device
    Ggen_vec_list = cuda.to_device(gen_vec_list)
    start_time_gpu = time.time()
    reads_vec_list = []
    for seq in reads:
        seq_vec = kmer(3,seq)
        reads_vec_list.append(seq_vec)
    Greads_vec_list = cuda.to_device(reads_vec_list)
    #calculate distance for each read
    threadperblock = 256
    blockpergrid = int(math.ceil(len(reads_vec_list)/threadperblock))
    minpos_list = cuda.device_array(len(reads_vec_list),dtype=int)
    #run gpu function
    cuda.synchronize()
    min_dist[blockpergrid,threadperblock](Ggen_vec_list,Greads_vec_list,minpos_list,len(reads_vec_list))
    cuda.synchronize()
    #copy result to host
    minpos_list = minpos_list.copy_to_host()
    minpos_list = minpos_list.tolist()
    print(minpos_list)
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu
    print("GPU process time: "+ str(gpu_time))





