import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
import itertools

cooffending_df = pd.read_csv('data/Cooffending.csv', na_values=['.'])
num_unique_crimes_per_mun = cooffending_df.groupby('MUN')['SeqE'].nunique()
num_mun = cooffending_df['MUN'].nunique()

mun_df = pd.read_csv('data/mun_clean.csv')
mun_codes_df = pd.read_csv('data/mun_codes.csv')

mun_df['admin_region'] = mun_df['admin_region'].apply(lambda x: int(x[-3:-1]))
cooffending_df = cooffending_df.loc[cooffending_df['annee'] == 2003]
mun_dict = dict(zip(mun_df.municipal_code,mun_df.admin_region))
cooffending_df['region'] = cooffending_df['MUN'].map(lambda x: mun_dict.get(x, np.NaN))
cooffending_df = cooffending_df.dropna()
cooffending_df['region'] = cooffending_df['region'].astype(int)
cooffending_df.to_csv('data/cooffending_regions.csv')

cooffending_df = pd.read_csv('data/cooffending_regions.csv')

region_adj_matrix = np.genfromtxt('data/region_adj.csv', delimiter=',')

'''
TODO: create graph G for with edges for neighboring municipalities
edges will have cost associated with them as specified
'''

# create dummy graph for testing
# G = nx.complete_graph(4)
# note that graph is 0 indexed but the regions are from 1
import time
start_time = time.time()

def get_route_edge_adj_matrix(G, source, target, cutoff=None):
	edges = list(G.edges())
	num_edges = G.number_of_edges()
	routes = list(nx.all_simple_paths(G, source, target, cutoff))
	num_routes = len(routes)

	def nodes_to_edge_list(route): return zip(route, route[1:])
	# list of lists of edge tuples for a route for s to t
	routes = map(nodes_to_edge_list, routes)

	# indices of adj matrix correspond to indices of routes and edges
	# in the routes and edges lists
	adj_matrix = np.zeros((num_routes, num_edges))

	for r in range(num_routes):
		for e in range(num_edges):
			if edges[e] in routes[r]:
				adj_matrix[r][e] = 1

	return adj_matrix


G = nx.from_numpy_matrix(region_adj_matrix)
G = G.to_directed()
nodes = list(G.nodes())
all_source_target_pairs = list(itertools.permutations(nodes, 2))

edges = list(G.edges())
edges = map(lambda x: ((x[0]+1, x[1]+1)), edges)

with open("edges.csv", "wb") as f:
    np.savetxt(f, edges, fmt='%i', delimiter=",")

for pair in all_source_target_pairs:
	source = pair[0]
	target = pair[1]
	adj_matrix = get_route_edge_adj_matrix(G, source, target)
	adj_matrix.astype(int)
	file_name = 'matrices/matrix_' + str(source+1) + '_' + str(target+1) + '.csv'
	with open(file_name, "wb") as f:
		np.savetxt(f, adj_matrix, fmt='%i', delimiter=",")

end_time = time.time()
print 'Finished creating matrix in', (end_time - start_time) / 60


