"""
Creates Web Matrices of magnitude 10 based on networkx code
"""

import networkx as nx
#from nx.utils import py_random_state
import random
import sys
import scipy.sparse
import numpy as np
import multiprocessing as mp
import timeit

#@py_random_state(7)
def web_graph(n,alpha=0.41,beta=0.54,gamma=0.05,delta_in=0.2,delta_out=0,create_using=None,seed=None):
  """
  [Include notes here]
  """
  def _choose_node(G, distribution, delta, psum):
    cumsum = 0.0
    #normalization
#    r = seed.random()
    r = random.uniform(0,1)
    for n,d in distribution:
      cumsum += (d+delta)/psum
      if r<cumsum:
        break
    return(n)

  if create_using is None or not hasattr(create_using, '_adj'):
    #start with 3-cycle
    G = nx.empty_graph(3, create_using=nx.DiGraph())
    G.add_edges_from([(0,1), (1,2), (2,0)])
  else:
    G = create_using
  if not G.is_directed():
    raise nx.NetworkXError("DiGraph required in create_using")

  if alpha <= 0:
    raise ValueError('alpha must be >= 0.')
  if beta <= 0:
    raise ValueError('beta must be >= 0.')
  if gamma <= 0:
    raise ValueError('gamma must be >=0.')

  if abs(alpha+beta+gamma - 1.0) >= 1e-9:
    raise ValueError('alpha+beta+gamma must be equal to 1.')

  number_of_edges = G.number_of_edges()
  while len(G) < n:
    psum_in = number_of_edges+delta_in*len(G)
    psum_out = number_of_edges+delta_out*len(G)
#    r = seed.random()
    r = random.uniform(0,1)
    #random choice in alpha,beta,gamma ranges
    if r<alpha:
      #alpha 
      #add new node v
      v = len(G)
      #choose w according to in-degree and delta_in
      w = _choose_node(G, G.in_degree(), delta_in, psum_in)
    elif r<alpha+beta:
      #beta
      #choose v according to out-degree and delta_out
      v = _choose_node(G, G.out_degree(), delta_out, psum_out)
      #choose w according to in-degree and delta_in
      w = _choose_node(G, G.in_degree(), delta_in, psum_in)
    else:
      #gamma
      #choose v according to out-degree and delta_out
      v = _choose_node(G, G.out_degree(), delta_out, psum_out)
      #add new node w
      w = len(G)

    #adjustment: no repeated edges (used digraph instead) and no loops
    if v != w:
      G.add_edge(v,w)
      number_of_edges += 1
  return(G)

def adjacency_matrix(G, nodelist=None, weight='weight'):
  """
  Adjusted from the package function so it goes from j to i
  Returns the sparse matrix in csr format
  """
  return nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, format='csc').transpose()

def initial_graph(index):
  G = nx.DiGraph()
  edgeList = list()
  if index == 0:
    edgeList = [(0,1),(1,2),(2,0)]
  elif index == 1:
    edgeList = [(0,3),(1,3),(2,3)]
  elif index == 2:
    edgeList = [(0,1),(1,3),(2,1),(1,2),(2,3),(3,0)]
  elif index == 3:
    edgeList = [(0,4),(1,0),(2,1),(3,4),(4,0),(4,1),(4,2),(4,3)]
  elif index == 4:
    edgeList = [(0,1),(1,2),(2,1),(2,0),(3,5),(4,3),(4,5),(5,4)]
  else:
    edgeList = [(0,1),(1,3),(2,0),(2,1),(3,2),(4,5),(6,4),(6,5),(7,8),(8,7)]
  G.add_edges_from(edgeList)
  return(G)

def saveData(index):
  G = initial_graph(index)
  for i in range(1,6): #might want to try 1,000,000 later
    num_vertices = 10**i
    print("Creating graph "+str(index)+" with 10^"+str(i)+" vertices...")
    tic = timeit.default_timer()
    G = web_graph(num_vertices, create_using=G)
    outName_graph = "data/graph_"+str(index)+"_1e"+str(i)+".web.gz"
    print("Time Elapsed: ",timeit.default_timer()-tic)
    print("Saving graph "+str(index)+" with 10^"+str(i)+" vertices...")
    nx.write_gpickle(G, outName_graph)
    print("Creating matrix "+str(index)+" with 10^"+str(i)+" vertices...")
    A_sparse = adjacency_matrix(G)
    outName_matrix = "data/matrix_"+str(index)+"_1e"+str(i)+".npz"
    print("Saving matrix "+str(index)+" with 10^"+str(i)+" vertices...")
    scipy.sparse.save_npz(outName_matrix, A_sparse)
    outName_matrix2 = "data/np_matrix_"+str(index)+"_1e"+str(i)+".npz"
    np.savez_compressed(outName_matrix2, A_sparse)


if __name__ == "__main__":
#  try:
#    maxNum = sys.argv[1]
#    maxNum = int(maxNum)
#  except:
#    maxNum = 1000000
  p = mp.Pool(6)
  p.map(saveData, range(6)) 
#  G = None
#  for i in range(1,6):
#    num_vertices = 10**i
#    print("Creating the graph with 10^"+str(i)+" vertices...") ##
#    G = web_graph(num_vertices, create_using=G)
#    outName_graph = "data/graph_1_1e"+str(i)+".web.gz"
#    print("Saving the graph with 10^"+str(i)+" vertices...") ##
#    nx.write_gpickle(G, outName_graph)
#    print("Creating the matrix for 10^"+str(i)+" vertices...") ##
#    A_sparse = adjacency_matrix(G)
#    outName_matrix = "data/matrix_1_1e"+str(i)+".npz"
#    print("Saving the matrix for 10^"+str(i)+" vertices...") ##
#    scipy.sparse.save_npz(outName_matrix,A_sparse)
#    outName_matrix2 = "data/matrix_1_1e"+str(i)+"_np.npz"
#    np.savez_compressed(outName_matrix2, A_sparse)

