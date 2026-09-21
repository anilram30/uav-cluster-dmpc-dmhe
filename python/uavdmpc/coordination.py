"""
coordination.py -- cluster graph, formation geometry, and the distributed
consensus/ADMM coordination that sits above the per-agent MPC/MHE.

  * Graph: adjacency + Laplacian, algebraic connectivity check.
  * Formation: fixed body-frame offsets d_i defining the desired shape.
  * Consensus: each agent holds a local estimate c_i of the formation-center
    reference and drives it to agreement over the graph (Olfati-Saber).
  * ADMM relative-agreement (optional): consensus on relative positions p_i-p_j.
"""
import numpy as np


class Cluster:
    def __init__(self, offsets, edges):
        """offsets: (Na,3) desired formation offsets; edges: list of (i,j)."""
        self.offsets = np.asarray(offsets, float)
        self.Na = len(self.offsets)
        self.edges = edges
        self.adj = np.zeros((self.Na, self.Na))
        for i, j in edges:
            self.adj[i, j] = self.adj[j, i] = 1.0
        self.deg = np.diag(self.adj.sum(1))
        self.L = self.deg - self.adj
        self.graph = {i: [j for j in range(self.Na) if self.adj[i, j] > 0]
                      for i in range(self.Na)}

    def algebraic_connectivity(self):
        ev = np.sort(np.linalg.eigvalsh(self.L))
        return ev[1]     # Fiedler value; >0 iff connected

    def desired_positions(self, center):
        return center[None, :] + self.offsets

    def _metropolis(self):
        deg = {i: len(self.graph[i]) for i in range(self.Na)}
        W = {}
        for i in range(self.Na):
            wij = {}; s = 0.0
            for j in self.graph[i]:
                w = 1.0/(1.0 + max(deg[i], deg[j])); wij[j] = w; s += w
            wij[i] = 1.0 - s
            W[i] = wij
        return W

    def consensus_center(self, c_local, rounds=3):
        """Drive per-agent center estimates c_local (Na,3) to agreement using
        Metropolis-Hastings weights (provably stable for any connected graph)."""
        W = self._metropolis()
        c = c_local.copy()
        for _ in range(rounds):
            newc = np.zeros_like(c)
            for i in range(self.Na):
                acc = W[i][i]*c[i]
                for j in self.graph[i]:
                    acc = acc + W[i][j]*c[j]
                newc[i] = acc
            c = newc
        return c
