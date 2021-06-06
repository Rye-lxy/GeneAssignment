import os


# Read short read file
def read_fa(file_name: str):
    seq_list = []
    seq = ''  # One short read sequence
    flag_new = False
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()

            if line:
                if line[0] == '>' and not flag_new:
                    seq_list.append(seq)
                    flag_new = True
                    continue
                elif flag_new:
                    seq = line[:-1]  # Delete escape character in each line
                    flag_new = False
                else:
                    seq += line[:-1]
            else:
                seq_list.append(seq)
                break

    seq_list = seq_list[1:]  # Delete the first empty string

    return seq_list


# Read one genome file
def read_fna(file_name: str):
    seq = ''  # Gene sequence
    desp = ''  # Description of the genome
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()

            if line:
                if line[0] == '>':
                    desp = line[1:-1]
                    continue
                else:
                    seq += line[:-1]  # Delete escape character in each line
            else:
                break

    return seq, desp


# Read all genome files with .fna suffix in 1 directory
def read_genome(dir_path: str = '.'):
    genome_list = []
    temp_dir = os.getcwd()

    os.chdir(dir_path)
    for root, dirs, files in os.walk(".", False):
        for file in files:
            if file.endswith('fna'):
                genome_list.append(read_fna(file))

    os.chdir(temp_dir)
    return genome_list


# Code the genome sequence with digit 0, 1, 2, 3.
def encoder(sequence: str):
    coded_seq = ''
    for char in sequence:
        if char == 'A' or char == 'a':
            coded_seq += '0'
        elif char == 'G' or char == 'g':
            coded_seq += '1'
        elif char == 'C' or char == 'c':
            coded_seq += '2'
        elif char == 'T' or char == 't':
            coded_seq += '3'
        else:
            continue

    return coded_seq


#Code the sequence with integrs for GPU functions
def encoder_int(sequence: str):
    coded_seq = []
    for char in sequence:
        if char == 'A' or char == 'a':
            coded_seq.append(int('0'))
        elif char == 'G' or char == 'g':
            coded_seq.append(int('1'))
        elif char == 'C' or char == 'c':
            coded_seq.append(int('2'))
        elif char == 'T' or char == 't':
            coded_seq.append(int('3'))
        else: #code randomly with 0,1,2,3
            coded_seq.append(random.randint(0,3))

    return coded_seq


# Get reference result
def get_ref(genome_list: list):
    ref_list = []

    with open('seq_id.map', 'r') as file:
        while True:
            line = file.readline()

            if line:
                gen = line.split(sep='\t')[-1][:-1]
                for i in range(len(genome_list)):
                    desp = genome_list[i][1].split('| ')[-1]
                    if desp == gen:
                        ref_list.append(i)
            else:
                break

    return ref_list


def accuracy(res: list, ref: list):
    # ref: Reference result
    # res: Calculated result
    if len(ref) == len(res):
        score: float = 0
        for i in range(len(ref)):
            if ref[i] == res[i]:
                score += 1.0

        return score / len(res)
    else:
        raise Exception('The length of ref (%d) and res (%d) are not equal!' % (len(ref), len(res)))

