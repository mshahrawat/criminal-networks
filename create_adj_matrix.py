import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx

cooffending_df = pd.read_csv('Cooffending.csv', na_values=['.'])
# number of unique crimes per municipality 
# print cooffending_df.groupby('MUN')['SeqE'].nunique()
num_mun = cooffending_df['MUN'].nunique()

'''
TODO: create graph G for with edges for neighboring municipalities
edges will have cost associated with them as specified
'''
# adj_list = []
# G = nx.Graph()
# G.add_edges_from(adj_list)

# create dummy graph for testing
G = nx.complete_graph(4)
source = 0
target = 3

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
