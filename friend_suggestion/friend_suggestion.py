#Uses python3

import sys
from collections import defaultdict
from queue import Queue
import math
import heapq

KEY = 1

class Heap():

    def __init__(self, data, size, key=None):
        self.data = data
        self.size = size
        self.swaps = []
        self.key = key

        if self.key:
           self.data = [(item, key(item)) for item in self.data]
    
    
    # for zero-based indexing
    # left_child of node[i] = 2i + 1
    # right child of node[i] = 2i + 2
    # parent of node[i] = round_up(i/2) - 1

    def get_parent(self, i):
        index = math.ceil(i/2) - 1
        return index if index > 0 and index < self.size else i

    def get_left_child(self, i):
        index = 2*i + 1
        return index if index < self.size else i

    def get_right_child(self, i):
        index = 2*i + 2
        return index if index < self.size else i

    def swap(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]
        self.swaps.append((i, j))

    def sift_down(self, i):
        # to sift a node down in min heap, check that it is strictly greater than any of its children. 
        # If both are >, swap it with the greater of the 2 children
        # If one of them are >, swap with that child
        # If neither are >, do nothing


        left_child_i = self.get_left_child(i)
        right_child_i = self.get_right_child(i)

        if self.key:

            global KEY
    
            node = self.data[i][KEY]
            left_child = self.data[left_child_i][KEY]
            right_child = self.data[right_child_i][KEY]

        else:
            node = self.data[i]
            left_child = self.data[left_child_i]
            right_child = self.data[right_child_i]

        if node > left_child and node > right_child:

            if left_child <= right_child:
                j =  left_child_i

            else:
                j = right_child_i

            self.swap(i, j)
            self.sift_down(j)
        
        elif node > left_child:
            j = left_child_i
            self.swap(i, j)
            self.sift_down(j)
        
        elif node > right_child:
            j = right_child_i
            self.swap(i, j)
            self.sift_down(j)

    
    # to build a min heap from an array, sift down all nodes from the top to second-last layer (nodes i = n/2 to 1, n = len(array))
    def build_heap(self):
        for i in range(self.size//2, -1, -1):
            self.sift_down(i)

    def build_heap_inf(self, s):
        self.swap(0, s)

    
    def insert(self, element):
        if self.key:
            self.data = [(element, self.key(element))] + self.data
        else:
            self.data = [element] + self.data

        self.sift_down(0)
        self.size += 1

    def extractMin(self):

        if self.key:
       
            min_item = self.data[0][0]

        else:

            min_item = self.data[0]

        self.data = self.data[1:]
        self.size -= 1

        if self.size > 0:
            self.sift_down(0)

        return min_item

    def isEmpty(self):
        return self.size == 0

class Vertex():
    def __init__(self, index):
        self.index = index

        # for computing of shortest path
        self.dist = math.inf

        # for computing layer of vertex in tree and retracing shortest path  
        self.prev = None

    def reset(self):
        self.__init__(self.index)


class Graph():
    def __init__(self, n, edges):
        
        self.vertices = [Vertex(i) for i in range(n)]
        
        self.edges = edges

        self.adj = defaultdict(list)

        self.last_processed = None

        self.known_region = []

        self.search_fin = False

        self.heap = None
        
        
        for ((u, v), w) in edges:

            self.adj[self.vertices[u-1]].append((self.vertices[v-1], w))



    def dijkstra(self):

        # use a binary min heap as priority queue for constant time extraction of min dist vertex

        # stop iterating once all vertices are of known distances
        if len(self.known_region) < len(self.vertices):

            u = self.heap.extractMin()
        
            self.last_processed = u

            if not u in self.known_region:
                self.known_region.append(u)

            for (v, w) in self.adj[u]:

                if v.dist > u.dist + w and not v in self.known_region:

                    v.dist = u.dist + w

                    # for backtracing of shortest path
                    v.prev = u

                    self.heap.insert(v)
        else:

            self.search_fin = True

  
class BiDijkstra():

    def __init__(self, graph, reverse_graph):
        self.graph = graph
        self.reverse_graph = reverse_graph

    def reset(self):
        for v in self.graph.vertices:
            v.dist = math.inf

        for v in self.reverse_graph.vertices:
            v.dist = math.inf

        self.reverse_graph.last_processed = None

        self.reverse_graph.known_region = []

        self.reverse_graph.search_fin = False

        self.reverse_graph.heap = None
        
        self.graph.last_processed = None

        self.graph.known_region = []

        self.graph.search_fin = False

        self.graph.heap = None


    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        if s == t:
            return 0
        
        self.reset()

        self.graph.vertices[s].dist = 0
        self.reverse_graph.vertices[t].dist = 0

        self.graph.heap = Heap(self.graph.vertices, len(self.graph.vertices), key=lambda x: x.dist)

        self.graph.heap.build_heap_inf(s)

        self.reverse_graph.heap = Heap(self.reverse_graph.vertices, len(self.reverse_graph.vertices), key=lambda x: x.dist)

        self.reverse_graph.heap.build_heap_inf(t)


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


    edges = []
    r_edges = []

    for edge in range(m):
        u, v, w = readl()
        edges.append(((u, v), w))
        r_edges.append(((v, u), w))


    num_queries, = readl()

    graph = Graph(n, edges)
    reverse_graph = Graph(n, r_edges)

    
    biDij = BiDijkstra(graph, reverse_graph)

    for i in range(num_queries):
        s, t = readl()
        s -= 1
        t -= 1

        dist = biDij.get_shortest_path(s, t) 

        ans = dist if not math.isinf(dist) else -1

        print(ans)


    