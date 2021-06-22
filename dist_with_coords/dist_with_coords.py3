#Uses python3

import sys
from collections import defaultdict
from queue import Queue
import math
import heapq


class Vertex():
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y

        # for computing of shortest path
        self.dist = math.inf
        self.pi = math.inf

    def __lt__(self, other):
        return self.pi < other.pi
    
    def calc_potential(self, t_x, t_y):
        pi = math.sqrt((self.x - t_x)**2 + (self.y - t_y)**2)
        self.pi = self.dist + pi


class Graph():

    def __init__(self, adj, vertices):

        # create a dict with vertex as key and list of its neighbours as values
        self.adj = adj
        self.vertices = vertices
        self.heap = None

    def reset(self, s, t):

        for u in self.vertices:
            u.dist = math.inf
 
        s = self.vertices[s]
        s.dist = 0

        self.heap = [s]

    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        self.reset(s, t)
        
        t_x = self.vertices[t].x
        t_y = self.vertices[t].y

        # stop iterating once all vertices are of known distances
        while len(self.heap) > 0:
        
            heapq.heapify(self.heap)
            u = heapq.heappop(self.heap)

            if u.index == t: return u.dist
                    
            for v, w in self.adj[u].items():

                if v.dist > u.dist + w:

                    v.dist = u.dist + w
                    v.calc_potential(t_x, t_y)

                    heapq.heappush(self.heap, v)

        return -1 if math.isinf(self.vertices[t].dist) else self.vertices[t].dist

def readl():
    return map(int, sys.stdin.readline().split())


if __name__ == '__main__':
    n, m = readl()
    vertices = []

    adj = defaultdict(dict)

    for i in range(n):
        x, y = readl()
        vertices.append(Vertex(i, x, y))
        
    for e in range(m):
        u,v,w = readl()
        adj[vertices[u-1]].update({vertices[v-1] : w})

    num_queries, = readl()

    graph = Graph(adj, vertices)

    for i in range(num_queries):

        s, t = readl()
        print(graph.get_shortest_path(s-1, t-1))

        
    ## STRESS TEST
    # with open('/Users/brandonthio/Python/Coursera Data Structures & Algorithms /week_6_advanced_shortest_path/dist_with_coords/tests/03.txt', 'r') as query:
    #     with open('/Users/brandonthio/Python/Coursera Data Structures & Algorithms /week_6_advanced_shortest_path/dist_with_coords/tests/03(a).txt', 'r') as solution:
    #         n, m = map(int, query.readline().split())
    #         vertices = []

    #         adj = defaultdict(dict)

    #         for i in range(n):
    #             x, y = map(int, query.readline().split())
    #             vertices.append(Vertex(i, x, y))
                
    #         for e in range(m):
    #             u, v, w = map(int, query.readline().split())
    #             adj[vertices[u-1]].update({vertices[v-1] : w})

    #         num_queries = int(query.readline())

    #         graph = Graph(adj, vertices)

    #         answers = []
    #         print('reading queries...')
    #         for i in range(num_queries):
                
    #             s, t = map(int, query.readline().split())
    #             print('source', s, 'sink', t)

    #             ans = graph.get_shortest_path(s-1, t-1)
    #             correct = int(solution.readline())

    #             if ans != correct:
    #                 print("INCORRECT")
    #                 print(ans, correct)
    #                 break

    #             else:
    #                 print("CORRECT")
    #                 print(ans)
    
