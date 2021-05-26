import numpy as np
from math import sqrt
from main import read_fa, read_fna, read_genome, encoder


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
def genome_assign(genome_list: list, seq_list: list):
    dist_list = []
    gen_mat = []

    # Get kmers in all genomes
    for genome in genome_list:
        gen = kmer(3, genome[0])
        gen_mat.append(gen)

    with open('result', 'w+') as file:
        for i in range(len(seq_list)):
            seq = seq_list[i]
            seq_vec = kmer(3, seq)

            for gen_vec in gen_mat:
                dist_list.append(get_dist(seq_vec, gen_vec))
                min_pos = dist_list.index(min(dist_list))

            # TODO: Determine a threshold of minimum distance

            res = "%d\t" % i + genome_list[min_pos][1]
            file.write(res)

    # Return the description of the genome
    # return genome_list[min_pos][1], dist_list
    return


if __name__ == '__main__':
    ls = [read_fna("NC_015656.fna")]
    # seq = 'AGGGGGCTGGCCCGTGACGAGCGGACCATCGTCGGCACCCCCGAGACGATCGCCGACCACATCCAGGAGTGG'
    # print(genome_assign(ls, seq))
