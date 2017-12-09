import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import networkx as nx

cooffending_df = pd.read_csv('Cooffending.csv', na_values=['.'])
num_unique_crimes_per_mun = cooffending_df.groupby('MUN')['SeqE'].nunique()
num_mun = cooffending_df['MUN'].nunique()

mun_df = pd.read_csv('mun_clean.csv')
mun_codes_df = pd.read_csv('mun_codes.csv')

mun_df['admin_region'] = mun_df['admin_region'].apply(lambda x: int(x[-3:-1]))
cooffending_df = cooffending_df.loc[cooffending_df['annee'] == 2003]
mun_dict = dict(zip(mun_df.municipal_code,mun_df.admin_region))
cooffending_df['region'] = cooffending_df['MUN'].map(lambda x: mun_dict.get(x, np.NaN))
cooffending_df = cooffending_df.dropna()
cooffending_df['region'] = cooffending_df['region'].astype(int)
cooffending_df.to_csv('cooffending_regions.csv')

cooffending_df = pd.read_csv('cooffending_regions.csv')

region_adj_matrix = np.genfromtxt('region_adj.csv', delimiter=',')

'''
TODO: create graph G for with edges for neighboring municipalities
edges will have cost associated with them as specified
'''

# create dummy graph for testing
# NEED directed graph with edges both ways
# G = nx.complete_graph(4)
# note that graph is 0 indexed but the regions are from 1
source = 3
target = 8
G = nx.from_numpy_matrix(region_adj_matrix)

def get_route_edge_adj_matrix(G, source, target, cutoff=None):
	edges = list(G.edges())
	num_edges = G.number_of_edges()
	routes = list(nx.all_simple_paths(G, source, target, cutoff))
	num_routes = len(routes)

	# lambda doesn't work correctly for some reason so have a function
	def nodes_to_edge_list(route): return zip(route, route[1:])
	# list of lists of edge tuples for a route for s to t
	routes = map(nodes_to_edge_list, routes)

	# indices of adj matrix correspond to indices of routes and edges
	# in the routes and edges lists
	adj_matrix = np.zeros((num_routes, num_edges))
	# filling in adj_matrix, not efficient but can fix later
	for r in range(num_routes):
		for e in range(num_edges):
			if edges[e] in routes[r]:
				adj_matrix[r][e] = 1

	return adj_matrix

print get_route_edge_adj_matrix(G, source, target)
