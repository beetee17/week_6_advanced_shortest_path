#Uses python3

import sys
from collections import defaultdict
from queue import Queue
import math
import heapq

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
    return arr

class Vertex():
    def __init__(self, index, dist=math.inf):
        self.index = index

        # for computing of shortest path
        self.dist = dist

        # for computing layer of vertex in tree and retracing shortest path  
        self.prev = None

    def __lt__(self, other):
        return self.dist < other.dist

    def reset(self):
        self.__init__(self.index)


class Graph():
    def __init__(self, vertices, edges, adj):
        
        self.vertices = vertices

        self.edges = edges

        self.adj = adj

        self.last_processed = None

        self.known_region = []

        self.search_fin = False

        self.heap = self.vertices[:]
        

    def reset(self):
        return self.__init__(self.vertices, self.edges, self.adj)



    def dijkstra(self):

        # use a binary min heap as priority queue for constant time extraction of min dist vertex

        # stop iterating once all vertices are of known distances
        if len(self.known_region) < len(self.vertices):

            u = heapq.heappop(self.heap)
        
            self.last_processed = u

            if not u in self.known_region:
                self.known_region.append(u)

            for (v, w) in self.adj[u]:

                if v.dist > u.dist + w and not v in self.known_region:

                    v.dist = u.dist + w

                    # for backtracing of shortest path
                    v.prev = u

                    heapq.heappush(self.heap, v)

                    heapq.heapify(self.heap)
        else:

            self.search_fin = True

  
class BiDijkstra():

    def __init__(self, graph, reverse_graph):
        self.graph = graph
        self.reverse_graph = reverse_graph

    def reset(self, s, t):
        

        self.graph.reset()
        self.reverse_graph.reset()

        # self.graph.vertices[s].dist = 0
        # self.reverse_graph.vertices[t].dist = 0

        for v in self.graph.vertices:
            if v.index == s:
                v.dist = 0
            else:
                v.dist = math.inf

        for v in self.reverse_graph.vertices:
            if v.index == t:
                v.dist = 0
            else:
                v.dist = math.inf

        self.graph.heap = swap(self.graph.heap, s, 0)
        self.reverse_graph.heap = swap(self.reverse_graph.heap, t, 0)



    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        if s == t:
            return 0
        
        self.reset(s, t)

        # heapq.heapify(self.graph.heap)
        # heapq.heapify(self.reverse_graph.heap)
        # print([(v.index+1, v.dist) for v in self.graph.vertices])
        # print([(v.index+1, v.dist) for v in self.reverse_graph.vertices])
        
        self.graph.dijkstra()
        self.reverse_graph.dijkstra()

        # print(self.graph.last_processed.index+1, self.reverse_graph.last_processed.index+1)

        while self.graph.last_processed.index != self.reverse_graph.last_processed.index:

            # print(self.graph.last_processed.index+1, self.reverse_graph.last_processed.index+1)
            # print([(v.index+1, v.dist) for v in self.graph.vertices])
            # print([(v.index+1, v.dist) for v in self.reverse_graph.vertices])

            # print(self.graph.search_fin, self.reverse_graph.search_fin)

            if self.graph.search_fin and self.reverse_graph.search_fin:
                return math.inf

            self.graph.dijkstra()

            self.reverse_graph.dijkstra()

            if self.graph.last_processed.index == t:
                # print('reach target')
                return self.graph.last_processed.dist 
            
            if self.reverse_graph.last_processed.index == s:
                # print('reach source')
                return self.reverse_graph.last_processed.dist 
            

        known_region = [u for u in self.graph.known_region if not math.isinf(self.reverse_graph.vertices[u.index].dist)] 

        best_dist = self.graph.last_processed.dist + self.reverse_graph.last_processed.dist
        best_u = self.graph.last_processed

        for u in known_region:
            curr_dist = self.graph.vertices[u.index].dist + self.reverse_graph.vertices[u.index].dist
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_u = u

        
        

        return best_dist    
            

        # return self.graph.last_processed.dist + self.reverse_graph.last_processed.dist

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

    vertices = [Vertex(i) for i in range(n)]
    r_vertices = [Vertex(i) for i in range(n)]

    edges = []
    r_edges = []

    adj = defaultdict(list)
    r_adj = defaultdict(list)

    for edge in range(m):
        u, v, w = readl()

        edges.append(((u, v), w))
        r_edges.append(((v, u), w))

        adj[vertices[u-1]].append((vertices[v-1], w))
        r_adj[r_vertices[v-1]].append((r_vertices[u-1], w))
    
    num_queries, = readl()

    graph = Graph(vertices, edges, adj)
    reverse_graph = Graph(r_vertices, r_edges, r_adj)

    
    biDij = BiDijkstra(graph, reverse_graph)

    for i in range(num_queries):
        s, t = readl()
        s -= 1
        t -= 1

        dist = biDij.get_shortest_path(s, t) 

        ans = dist if not math.isinf(dist) else -1

        print(ans)


    