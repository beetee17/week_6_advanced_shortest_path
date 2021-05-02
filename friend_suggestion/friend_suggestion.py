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

        # for computing layer of vertex in tree and retracing shortest path  
        self.prev = None


    def __lt__(self, other):
        return self.dist < other.dist


class Graph():
    def __init__(self, edges, adj):
        
        self.vertices = {}

        self.edges = edges

        self.adj = adj

        self.last_processed = None

        self.search_fin = False

        self.heap = None
        

    def reset(self, s):

        self.vertices = {s : 0}
      
        self.heap = [Vertex(s, dist=0)]
      
        self.last_processed = None

        self.search_fin = False



    def dijkstra(self):

        # use a binary min heap as priority queue for constant time extraction of min dist vertex

        # stop iterating once all vertices are of known distances


        if len(self.heap) > 0:

            u = heapq.heappop(self.heap)

            if u.dist != self.vertices.get(u.index, math.inf):
                return
            
            u = u.index
        
            self.last_processed = u


            for (v, w) in self.adj[u]:
                
                if self.vertices.get(v, math.inf) > self.vertices.get(u, math.inf) + w:

                    self.vertices[v] = self.vertices[u] + w

                    heapq.heappush(self.heap, Vertex(v, dist=self.vertices[v]))

        else:

            self.search_fin = True

  
class BiDijkstra():

    def __init__(self, graph, reverse_graph):
        self.graph = graph
        self.reverse_graph = reverse_graph


    def reset(self, s, t):
        
        self.graph.reset(s)
        self.reverse_graph.reset(t)

    def get_best_dist(self):
        
        best_dist = self.graph.vertices[self.graph.last_processed] + self.reverse_graph.vertices[self.reverse_graph.last_processed]
        best_u = self.graph.last_processed

        for u in self.graph.vertices.keys():

            curr_dist = self.graph.vertices[u] + self.reverse_graph.vertices.get(u, math.inf)

            if curr_dist < best_dist:

                best_dist = curr_dist
                best_u = u

            try:
                del self.reverse_graph.vertices[u]

            except KeyError:
                pass
        
        if self.reverse_graph.vertices:

            for u in self.reverse_graph.vertices.keys():

                curr_dist = self.graph.vertices.get(u, math.inf) + self.reverse_graph.vertices[u]

                if curr_dist < best_dist:
                    best_dist = curr_dist
                    best_u = u
                    
        return best_dist

    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        if s == t:
            return 0
        
        self.reset(s, t)

        self.graph.dijkstra()
        self.reverse_graph.dijkstra()


        while self.graph.last_processed != self.reverse_graph.last_processed:


            if self.graph.search_fin and self.reverse_graph.search_fin:
    
                return math.inf

            self.graph.dijkstra()

            self.reverse_graph.dijkstra()

            if self.graph.last_processed == t:
              
                return self.graph.vertices[t] 
            
            if self.reverse_graph.last_processed == s:
       
                return self.reverse_graph.vertices[s] 


        return self.get_best_dist()  
            



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

    edges = []
    r_edges = []

    adj = defaultdict(list)
    r_adj = defaultdict(list)

    for edge in range(m):
        u, v, w = readl()

        edges.append(((u, v), w))
        r_edges.append(((v, u), w))

        adj[u].append((v, w))
        r_adj[v].append((u, w))
    
    num_queries, = readl()

    graph = Graph(edges, adj)
    reverse_graph = Graph(r_edges, r_adj)

    
    biDij = BiDijkstra(graph, reverse_graph)

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


    