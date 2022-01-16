from itertools import combinations
import numpy as np
import networkx as nx


def generate_split_candidates(nodes):
    candidate_list = []
    for i in range(1, len(nodes) // 2 + 1):
        for a in combinations(nodes, i):
            a = set(a)
            b = nodes - a
            if len(a) == len(b):
                if min(a) < min(b):
                    candidate_list.append((a, b))
            else:
                candidate_list.append((a, b))
    return candidate_list


def split_graph(G, split_list):
    G_list = [G.subgraph(a) for a in split_list]
    G_r = nx.empty_graph()
    for g in G_list:
        G_r = nx.union(G_r, g)
    return G_r


def generate_masks(nodes):
    candidate_list = generate_split_candidates(nodes)
    G = nx.complete_graph(nodes)
    mask_list = []
    for split_list in candidate_list:
        G_r = split_graph(G, split_list)
        adj = nx.adjacency_matrix(G_r, nodelist=nodes).todense()
        mask = adj + np.identity(len(nodes))
        mask_list.append(mask)
    return mask_list, candidate_list
