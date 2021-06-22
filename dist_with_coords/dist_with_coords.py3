#!/usr/bin/python3

import sys
import queue
import math

# A* = Dijkstra but with potential function
# Bidirectional search needs two potential functions:
# πf (v): estimate on dist(v, t).
# πr(v): estimate on dist(s, v).
# Reduced cost of an arc (u, v):
# – Forward: new_arc(u, v) = arc(u, v) − πf (u) + πf (v).
# – Reverse: new_arc(v, u) = arc(v, u) − πr(v) + πr(u).
# In general, two arbitrary feasible functions πf and πr are not consistent.
# Their average is both feasible and consistent:
# pf(v) = 1/2 (πf (v) − πr(v))
# pr(v) = − pf(v)
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
        self.potential = math.inf
    
    def calc_potential(self, s_x, s_y, t_x, t_y, direction):
        # πf (v): estimate on dist(v, t).
        # πr(v): estimate on dist(s, v).
        
        pi_f = ((self.x-t_x)**2 + (self.y-t_y)**2)**0.5
        pi_r = ((s_x - self.x)**2 + (s_y - self.y)**2)**0.5
        pi = ((s_x - t_x)**2 + (s_y - t_y)**2)**0.5

        print('node', self.index, 'pi_f', pi_f, 'pi_r', pi_r, 'direction', direction)

        # pf(v) = 1/2 (πf (v) − πr(v))
        
        if direction == 'forward':
            self.potential = (pi_f - pi_r)/2 + pi/2
        
        # pr(v) = − pf(v)
        else:
            self.potential = (pi_r - pi_f)/2 + pi/2
        
        print('potential', self.potential)
        return self.potential

    def __lt__(self, other):
        return self.potential < other.potential
    

class Graph():

    def __init__(self, adj, x, y, direction):
        
        self.vertices = {}
        self.x = x
        self.y = y
        self.adj = adj
        self.processed = set()
        self.search_fin = False
        self.heap = None
        self.direction = direction
        

    def reset(self, s, t):

        self.vertices = {s : 0}
        v_s = Vertex(s, self.x[s], self.y[s])
        v_s.calc_potential(self.x[s], self.y[s], self.x[t], self.y[t], self.direction)
        self.heap = [v_s]
        self.processed = set()
        self.search_fin = False


    def get_next_vertex(self):

        # use a binary min heap as priority queue for constant time extraction of min dist vertex

        if len(self.heap) > 0: 
            
            u = heapq.heappop(self.heap)

            if not u.index in self.processed: return u
        
        else: self.search_fin = True

        
class AStar():

    def __init__(self, graph, reverse_graph):
        self.graph = graph
        self.reverse_graph = reverse_graph
        self.min_dist = math.inf
        self.s_cor = None
        self.t_cor = None
        self.potential_t = None


    def reset(self, s, t):
        
        self.graph.reset(s, t)
        self.reverse_graph.reset(t, s)
        self.min_dist = math.inf
        self.s_cor = (self.graph.x[s], self.graph.y[s])
        self.t_cor = (self.graph.x[t], self.graph.y[t])

        t = Vertex(t, self.graph.x[t], self.graph.y[t])
        self.potential_t = t.calc_potential(self.s_cor[0], self.s_cor[1], self.t_cor[0], self.t_cor[1], 'reverse')

    def process(self, u, s, t, direction):

        if not u: return None
        
        if direction == 'forward':

            graph = self.graph
            reverse_graph = self.reverse_graph
        
        else:

            graph = self.reverse_graph
            reverse_graph = self.graph

        graph.processed.add(u.index)

        print(graph.processed, reverse_graph.processed)

        print('checking node', u.index, u.potential)
        for (v, w) in graph.adj[u.index]:
            # Reduced cost/weight of an arc (u, v):
            # – Forward: new_weight(u, v) = weight(u, v) − pf(u) + pf (v).
            # – Reverse: new_weight(v, u) = weight(v, u) − pr(v) + pr(u).
            v = Vertex(v, graph.x[v], graph.y[v])
            v.calc_potential(self.s_cor[0], self.s_cor[1], self.t_cor[0], self.t_cor[1], direction)
            print('neighbour', v.index, v.potential, 'cost', w)
            w = w - u.potential + v.potential
            print('new cost', w)
            if graph.vertices.get(v.index, math.inf) > graph.vertices.get(u.index, math.inf) + w:

                graph.vertices[v.index] = graph.vertices[u.index] + w
                v.potential = graph.vertices[u.index] + w

                heapq.heappush(graph.heap, v)

            # In addition to relaxing edges outgoing from u,
            # we must maintain the length of the best path seen so far in min_dist
            # initially, min_dist = ∞;
            # when scanning an edge/arc (u, v) with weight, w, in the forward search and v is scanned in
            # the reverse search, update min_dist if forward_dist(u) + w + reverse_dist(v) < µ.
            # similar procedure if scanning an arc in the reverse search.
            
            if v.index in reverse_graph.processed:
                
                self.min_dist = min(self.min_dist, u.potential + w + reverse_graph.vertices[v.index])
                print('new min dist', self.min_dist)

    def get_shortest_path(self, s, t):
        """Given the index of a source vertex, compute the shortest path lengths of all vertices reachable from the source"""

        if s == t: return 0
        
        self.reset(s, t)

        while True:

            if self.graph.search_fin and self.reverse_graph.search_fin: 
                print('search fin')
                return math.inf
            print([v.index for v in self.graph.heap])
            print([v.index for v in self.reverse_graph.heap])

            u_forward = self.graph.get_next_vertex()
            u_reverse = self.reverse_graph.get_next_vertex()
            if u_forward:
                u_forward.calc_potential(self.s_cor[0], self.s_cor[1], self.t_cor[0], self.t_cor[1], 'forward')
            if u_reverse:
                u_reverse.calc_potential(self.s_cor[0], self.s_cor[1], self.t_cor[0], self.t_cor[1], 'reverse')
            # see https://www.cs.princeton.edu/courses/archive/spr06/cos423/Handouts/EPP%20shortest%20path%20algorithms.pdf 
            # for explanation of stopping criterion
            if u_forward and u_reverse and u_forward.potential + u_reverse.potential >= self.min_dist: 
                print('stop criteria')
                return self.min_dist

            if u_forward and u_forward.index == t: 
                print('reach target')
                return self.min_dist
            if u_reverse and u_reverse.index == s: 
                print('reach source')
                return self.min_dist

            self.process(u_forward, s, t, direction = 'forward')
            self.process(u_reverse, s, t, direction = 'reverse')



# 4 4
# 0 0 
# 0 1
# 2 1
# 2 0
# 1 2 1
# 4 1 2
# 2 3 2
# 1 3 6
# 1
# 1 3

def readl():
    return map(int, sys.stdin.readline().split())

    


if __name__ == '__main__':
    n, m = readl()
    x = [0 for _ in range(n)]
    y = [0 for _ in range(n)]
    adj = defaultdict(list)
    r_adj = defaultdict(list)

    for i in range(n):
        a, b = readl()
        x[i] = a
        y[i] = b

    for e in range(m):
        u,v,w = readl()
        adj[u-1].append((v-1, w))
        r_adj[v-1].append((u-1, w))

    num_queries, = readl()

    graph = Graph(adj, x, y, 'forward')
    reverse_graph = Graph(r_adj, x, y, 'reverse')
    astar = AStar(graph, reverse_graph)

    for i in range(num_queries):

        s, t = readl()
        dist = astar.get_shortest_path(s-1, t-1) 

        ans = dist if not math.isinf(dist) else -1
        print(ans)
