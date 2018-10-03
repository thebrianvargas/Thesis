"""
My Ghetto implementation of a graph to see if it can handle more memory
"""

import timeit, sys

class digraph:
  """
  Handles the storage of edges, vertices, and degrees in a directed graph
  """
  def __init__(self, edgeList=dict(), degrees=dict()):
    #the inputs must be dictionaries; most of the time will be empty anyway
    self.e = edgeList
    #the key is the tail the values are the heads
    self.d = degrees
    #the key is the vertex; the values are the degrees [din,dout]

  def add(self, tail, head):
    #if the vertex is new, construct it
    if tail not in self.e.keys():
      self.e[tail] = set()
      self.d[tail] = [0,0]
    if head not in self.e.keys():
      self.e[head] = set()
      self.d[head] = [0,0]
    #update corresponding counts
    self.d[tail][1] += 1
    self.d[head][0] += 1
    #make the connection from tail to vertex
    self.e[tail].add(head)

  def print_degrees(self):
    #print out the degrees
    print("Printing the in-degrees")
    for i in range(len(self.d.keys())):
      print(i,":",self.d[i][0])
    print("Printing the out-degrees")
    for i in range(len(self.d.keys())):
      print(i,":",self.d[i][1])

  def print_edges(self):
    #print out the edge list
    print("Printing the edge list")
    for i in range(len(self.e.keys())):
      print(i,":",self.e[i])

if __name__ == "__main__":
  zeros = int(sys.argv[1])
  tic = timeit.default_timer()
  G = digraph()
  for i in range(10**zeros):
    G.add(i,i+1)
  print("Graph is completed..")
  toc = timeit.default_timer()
  print("Time elapsed: ",toc-tic) 
