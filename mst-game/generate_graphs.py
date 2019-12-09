import time
import argparse
import pprint as pp
import os
import pickle

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

class Args():
	def __init__(self, args):
		self.graph_type = args.graph_type #'random_geometric_graph'
		self.num_graphs = args.num_graphs #10
		self.num_nodes = args.num_nodes #10
		self.prob = 0.5
		self.use_random_weights = args.use_random_weights #Default: False
		self.weight_fn = np.random.random
		self.get_spanning_trees = args.get_spanning_trees #True
		self.write_edges = args.write_edges #True
		self.filename = args.filename
		if self.filename is None:
			self.filename = f"./data/mst{self.num_nodes}_{self.num_graphs}_{self.graph_type}.txt"
		pp.pprint(vars(self))

def create(args):
	graphs = []
	poses = []
	#distances = []
	if args.graph_type == 'fast_gnp_random_graph':
		for i in range(args.num_graphs):
			graphs.append(nx.fast_gnp_random_graph(args.num_nodes, args.prob))

	elif args.graph_type == 'random_geometric_graph':
		for i in range(args.num_graphs):
			G = nx.random_geometric_graph(args.num_nodes, 2, 2)
			pos = nx.random_layout(G)
			positions = np.array([pos[n] for n in pos])
			dist = get_distances(positions, args.num_nodes)
			G.add_weighted_edges_from(dist)
			graphs.append(G)
			poses.append(pos)
			#distances.append(dist)

	if args.use_random_weights:
		add_edge_weights(graphs, args.weight_fn)

	spanning_trees = []
	if args.get_spanning_trees:
		for G in graphs:
			T = nx.minimum_spanning_tree(G)
			spanning_trees.append(T)

	if args.write_edges and args.get_spanning_trees:
		write_edges(graphs, spanning_trees, poses, filename=args.filename)
	return graphs, spanning_trees, poses#, distances


def get_generated_weights(graphs, num_nodes):
	graph_weights = []
	for G in graphs:
		edge_weights = [np.inf]*(num_nodes * num_nodes)
		for (u,v,weight) in G.edges.data():
			weight = np.round(weight['weight'], 4)
			edge_weights[u * num_nodes + v] = weight
			edge_weights[v * num_nodes + u] = weight
		generated_weights = ",".join(str(e) for e in edge_weights)
		graph_weights.append(generated_weights)
	return graph_weights

def get_solutions(graphs, num_nodes):
	MST_rewards = []
	MSTS_edges = []
	for T in graphs:
		MST_sum = 0
		MST_edges = []
		for (u,v,weight) in T.edges.data():
			weight = weight['weight']
			edge_1, edge_2 = u * num_nodes + v, v * num_nodes + u
			MST_sum += weight
			MST_edges.append((edge_1, edge_2, weight))
		MST_rewards.append(MST_sum)
		MST_edges.append(MST_edges)
	return MST_rewards, MST_edges

def save_game_data(num_graphs, num_nodes, fname=None):
	args_list = ['--num_graphs',str(num_graphs),'--num_nodes',str(num_nodes), '--write_edges', False]
	args = get_args(args_list)

	with open(fname, 'wb') as handle:
		for i in range(num_graphs):
			graphs, msts, _ = create(args)
			graph_inputs = get_generated_weights(graphs, num_nodes)
			graph_rewards, graph_edges = get_solutions(msts, num_nodes)
			data = {'inputs': graph_inputs,
					'solutions': graph_edges,
					'rewards': graph_rewards,
					'num_nodes': num_nodes,
					'num_graphs': num_graphs}
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return True

def generate_game_data(num_graphs, num_nodes, save=False, fname=None):
	args_list = ['--num_graphs',str(num_graphs),'--num_nodes',str(num_nodes), '--write_edges', False]
	args = get_args(args_list)
	graphs, msts, _ = create(args)
	
	graph_inputs = get_generated_weights(graphs, num_nodes)
	graph_rewards, graph_edges = get_solutions(msts, num_nodes)
	
	data = {'inputs': graph_inputs,
	 'solutions': graph_edges,
	 'rewards': graph_rewards,
	 'num_nodes': num_nodes,
	 'num_graphs': num_graphs}

	if save:
		with open(fname, 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return data

def write_edges(graphs, spanning_trees, poses, filename):
	with open(filename, 'w') as f:
		start_time = time.time()
		for G, T, pos in zip(graphs, spanning_trees, poses):
			keys = sorted(pos.keys())
			points = list(np.array([pos[key] for key in keys]).flatten())
			solution = T.edges()
			f.write( " ".join( str(x) for x in points) )
			f.write( str(" ") + str('output') + str(" ") )
			f.write( str(" ").join( str(edge) for edge in solution) )
			f.write( "\n" )
		end_time = time.time() - start_time

	print(f"Completed generation of {len(graphs)} samples of MST{len(pos.keys())}.")
	print(f"Total time: {end_time/3600:.1f}h")
	print(f"Average time: {(end_time/3600)/len(graphs):.1f}h")


def plot(graphs, spanning_trees = None, poses = None):
	if not poses:
		poses = [None]*len(graphs)

	if not spanning_trees:
		spanning_trees = [None]*len(graphs)
	
	assert len(spanning_trees) == len(graphs) == len(poses)
	for (G,T,pos) in zip(graphs, spanning_trees, poses):
		plot_single(G,T,pos)

	#plt.savefig('./graphs-spanning_tree.png')

def plot_single(G,T=None,pos=None):
	if not pos:
		pos = nx.spring_layout(G)
	nx.drawing.nx_pylab.draw_networkx(G, pos=pos)
	plt.show()
	if T:
		nx.drawing.nx_pylab.draw_networkx(T, pos=pos)
		plt.show()


def add_edge_weights(graphs, weight_fn = np.random.random):
	for G in graphs:
		for (u,v,weight) in G.edges(data=True):
			G.add_edge(u,v, weight=weight_fn())

def get_distances(x, num_nodes):
	dist = cdist(x,x, metric='euclidean')
	return [(i,j,dist[i,j]) for i in range(num_nodes) for j in range(i)] # ebunch


def get_args(params):
	return Args(get_args_from_parser(params))

def get_args_from_parser(otherArgs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--graph_type", type=str, default='random_geometric_graph')
	parser.add_argument("--num_graphs", type=int, default=10000)
	parser.add_argument("--num_nodes", type=int, default=20)
	parser.add_argument("--use_random_weights",'-w',type=bool, default=False)
	parser.add_argument("--get_spanning_trees",type=bool, default=True)
	parser.add_argument("--write_edges", type=bool, default=True)
	parser.add_argument("--filename", type=str, default=None)
	return parser.parse_args(otherArgs)

def main():
	opts = get_args_from_parser(None)
	args = Args(opts)
	graphs, msts, poses, distances = create(args) # creates graphs and writes to file

if __name__ == "__main__":
	main()
