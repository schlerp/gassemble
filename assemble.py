from typing import List, Tuple
import os
import random
import numpy as np
from tqdm import tqdm
from numba import njit, jit
import networkx as nx
from matplotlib import pyplot as plt
from multiprocessing import Pool


BASE_NUCL_MAP = {
    0: "A",
    1: "C",
    2: "G",
}

DNA_NUCL_MAP = {**BASE_NUCL_MAP, 3: "T"}

RNA_NUCL_MAP = {**BASE_NUCL_MAP, 3: "U"}


def bases_as_text(base_list: List[int], is_rna: bool = False) -> List[str]:
    """converts a list of ints into a list of strings representing the nuleotides"""
    if is_rna:
        return [RNA_NUCL_MAP[x] for x in base_list]
    return [DNA_NUCL_MAP[x] for x in base_list]


def create_dataset(
    genome_len: int = 6000, n_reads: int = 100, read_size: int = 100
) -> Tuple[List[int], List[List[int]]]:
    """builds an example genome, returns it and a set of reads that make it up"""
    print(
        f"creating dataset with genome length: {genome_len}, with {n_reads} reads of size {read_size}..."
    )
    genome = np.random.randint(4, size=genome_len, dtype="u1")
    reads = np.zeros(shape=(n_reads, read_size), dtype="u1")
    for read_id in tqdm(range(n_reads)):
        start_idx = random.choice(range(0, genome_len - read_size + 1))
        reads[read_id] = genome[start_idx : start_idx + read_size]
    return genome, reads


def create_kmers(reads: List[List[int]], k=5) -> List[List[int]]:
    """breaks each read into a list of k-mers"""
    kmer_per_read = read_size - k + 1
    total_kmers = n_reads * kmer_per_read
    print(
        f"creating {kmer_per_read} kmers per read of length {k} for total of {total_kmers}..."
    )
    kmers = np.zeros(shape=(n_reads * kmer_per_read, k), dtype="u1")
    for read_idx, read in tqdm(enumerate(reads), total=len(reads)):
        for kmer_idx in range(kmer_per_read):
            # for zero based counting *facepalm*
            kmers[((read_idx + 1) * (kmer_idx + 1)) - 1] = read[kmer_idx : kmer_idx + k]
    return kmers


@njit
def equal_all(x: np.array, y: np.array) -> bool:
    return np.all(np.equal(x, y))


@njit
def left_overlap(kmer: np.array) -> np.array:
    return kmer[1:]


@njit
def right_overlap(kmer: np.array) -> np.array:
    return kmer[:-1]


@njit
def debruijn_edges_for_kmer(kmer: np.array, kmers: np.array) -> List[List[np.array]]:
    edges = []
    left1 = left_overlap(kmer)
    right1 = right_overlap(kmer)
    for kmer2 in kmers:
        left2 = left_overlap(kmer2)
        right2 = right_overlap(kmer2)
        # if they arent the same kmer
        if not equal_all(kmer, kmer2):
            # if they do overlap
            if equal_all(left1, right2):
                edges.append([right1, right2])
                # edges.append([kmer, kmer2])
            if equal_all(right1, left2):
                edges.append([left2, left1])
                # edges.append([kmer2, kmer])
    return edges


def create_debruijn_edges(kmers: np.array):
    """creates a de-bruijn edge list of the k-mers"""
    print("creating de Bruijn edges...")
    pool = Pool(4)
    responses = []
    responses = pool.starmap(
        debruijn_edges_for_kmer, ((kmer_left, kmers) for kmer_left in kmers)
    )

    pool.close()
    pool.join()

    edges = []
    for response in responses:
        edges.extend(response)

    return edges


def to_2bit_string(x: int):
    if x > 3 or x < 0:
        raise Exception("number was too small/big for two bits")
    return bin(x)[2:].zfill(2)


def to_bin(base_list):
    return "".join([to_2bit_string(base) for base in base_list])


def convert_edges_to_python(
    edges: List[List[np.array]], as_text: bool = False
) -> List[Tuple[int]]:
    print("converting edges back into native python objects...")
    new_edges = []
    for source, target in tqdm(edges):
        if as_text:
            new_edges.append(
                (
                    "".join(bases_as_text(source.tolist())),
                    "".join(bases_as_text(target.tolist())),
                )
            )
        else:
            new_edges.append((to_bin(source.tolist()), to_bin(target.tolist())))
    return new_edges


def write_edges(fpath, edges):
    with open(fpath, "w+") as f:
        for source, target in edges:
            f.write(f"{source}\t{target}\n")


def reconstruct_sequence(eulerian_path: List):
    sequence_string = eulerian_path.pop(0)
    for kmer in eulerian_path:
        sequence_string += kmer[-1]
    return sequence_string


if __name__ == "__main__":
    sequence_size = 100
    read_size = 10
    n_reads = 1_000
    sequence, reads = create_dataset(sequence_size, n_reads, read_size)

    kmers = create_kmers(reads, 6)

    print("sequence:", "".join(bases_as_text(sequence)))
    # [print("read:", "".join(bases_as_text(x))) for x in reads]
    # print(["".join(bases_as_text(x)) for x in kmers])
    # print("creating debruin edges")

    edges = create_debruijn_edges(kmers)

    py_edges = convert_edges_to_python(edges, as_text=True)

    write_edges("./edges.txt", py_edges)

    print("constructing graph...")
    g = nx.MultiDiGraph(py_edges, name=sequence)

    eulerian_path = [x for x in g]

    print("eulerian path:", eulerian_path)

    reconstructed_seq = reconstruct_sequence(eulerian_path)

    print("reconstructed sequence:", reconstructed_seq)

    if reconstructed_seq == "".join(bases_as_text(sequence)):
        print("Success, sequences match!")
    else:
        print("FAILURE!")

    print("done!")
