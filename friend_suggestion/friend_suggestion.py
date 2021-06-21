#Uses python3

import sys
from collections import defaultdict
from queue import Queue
import math
import heapq


class Vertex():
    def __init__(self, index, dist=math.inf):
        self.index = index

        # for computing of shortest path
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist


class Graph():
    def __init__(self, adj):
        
        self.vertices = {}
        self.adj = adj
        self.processed = set()
        self.search_fin = False
        self.heap = None
        

    def reset(self, s):

        self.vertices = {s : 0}
        self.heap = [Vertex(s, dist=0)]
        self.processed = set()
        self.search_fin = False


    def get_next_vertex(self):

        # use a binary min heap as priority queue for constant time extraction of min dist vertex

        if len(self.heap) > 0: 
            
            u = heapq.heappop(self.heap)

            if u.dist == self.vertices.get(u.index, math.inf): return u
        
        else: self.search_fin = True

        
class BiDijkstra():

    def __init__(self, graph, reverse_graph):
        self.graph = graph
        self.reverse_graph = reverse_graph
        self.min_dist = math.inf


    def reset(self, s, t):
        
        self.graph.reset(s)
        self.reverse_graph.reset(t)
        self.min_dist = math.inf

    def process(self, u, s, t, direction):

        if not u: return None

        if direction == 'forward':

            if u.index == t: return u.dist
            
            graph = self.graph
            reverse_graph = self.reverse_graph
        
        else:

            if u.index == s: return u.dist 

            graph = self.reverse_graph
            reverse_graph = self.graph

        graph.processed.add(u.index)

        for (v, w) in graph.adj[u.index]:
    
            if graph.vertices.get(v, math.inf) > graph.vertices.get(u.index, math.inf) + w:

                graph.vertices[v] = graph.vertices[u.index] + w

                heapq.heappush(graph.heap, Vertex(v, dist=graph.vertices[v]))

            # In addition to relaxing edges outgoing from u,
            # we must maintain the length of the best path seen so far in min_dist
            # initially, min_dist = ∞;
            # when scanning an edge/arc (u, v) with weight, w, in the forward search and v is scanned in
            # the reverse search, update min_dist if forward_dist(u) + w + reverse_dist(v) < µ.
            # similar procedure if scanning an arc in the reverse search.
            
            if v in reverse_graph.processed:

                self.min_dist = min(self.min_dist, u.dist + w + reverse_graph.vertices[v])

    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        if s == t: return 0
        
        self.reset(s, t)

        while True:

            if self.graph.search_fin and self.reverse_graph.search_fin: return math.inf

            u_forward = self.graph.get_next_vertex()
            u_reverse = self.reverse_graph.get_next_vertex()
            
            # see https://www.cs.princeton.edu/courses/archive/spr06/cos423/Handouts/EPP%20shortest%20path%20algorithms.pdf 
            # for explanation of stopping criterion
            if u_forward and u_reverse and u_forward.dist + u_reverse.dist >= self.min_dist: return self.min_dist

            self.process(u_forward, s, t, direction = 'forward')
            self.process(u_reverse, s, t, direction = 'reverse')

            




# 5 20
# 1 2 667
# 1 3 677
# 1 4 700
# 1 5 622
# 2 1 118
# 2 3 325
# 2 4 784
# 2 5 11
# 3 1 585
# 3 2 956
# 3 4 551
# 3 5 559
# 4 1 503
# 4 2 722
# 4 3 331
# 4 5 366
# 5 1 880
# 5 2 883
# 5 3 461
# 5 4 228
# 10
# 1 1
# 1 2
# 1 3
# 1 4
# 1 5
# 2 1
# 2 2
# 2 3
# 2 4
# 2 5
       

# 0
# 667
# 677
# 700
# 622
# 118
# 0
# 325
# 239
# 11



def readl():
    return map(int, sys.stdin.readline().split())

if __name__ == '__main__':

    n, m = readl()

    adj = defaultdict(list)
    r_adj = defaultdict(list)

    for edge in range(m):
        u, v, w = readl()
        adj[u].append((v, w))
        r_adj[v].append((u, w))
    
    num_queries, = readl()

    biDij = BiDijkstra(Graph(adj), Graph(r_adj))

    for i in range(num_queries):
        s, t = readl()
        dist = biDij.get_shortest_path(s, t) 

        ans = dist if not math.isinf(dist) else -1
        print(ans)


    

## STRESS TEST ###
    # import random
    # import time

    # n = 1000
    # m = 100000

    # edges = []
    # r_edges = []

    # adj = defaultdict(list)
    # r_adj = defaultdict(list)

    # for edge in range(m):
    #     if edge % 100000 == 0:
    #         print(edge)
    #     u = random.randint(0, n)
    #     v = random.randint(0, n)

    #     while v == u:

    #         v = random.randint(0, n)

    #     w = 1

    #     edges.append(((u, v), w))
    #     r_edges.append(((v, u), w))

    #     adj[u].append((v, w))
    #     r_adj[v].append((u, w))
    

    # num_queries = 1000

    # graph = Graph(edges, adj)
    # reverse_graph = Graph(r_edges, r_adj)

    
    # biDij = BiDijkstra(graph, reverse_graph)

    # start = time.time()
    # for i in range(num_queries):
        
        
    #     s = random.randint(0, n)
    #     t = random.randint(0, n)

    #     q_start = time.time()
    #     dist = biDij.get_shortest_path(s, t) 

    #     ans = dist if not math.isinf(dist) else -1

    #     q_end = time.time()
    #     print('query time {}'.format(q_end-q_start))
        
    
    # end = time.time()
    # print("total time: {}".format(end-start))


    