"""
Phase 1 — Memristor crossbar simulation
Run standalone: python phase1_crossbar/crossbar.py
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

G = np.array([
    [0.8, 0.2, 0.9, 0.1],
    [0.3, 0.7, 0.4, 0.6],
    [0.5, 0.5, 0.8, 0.2],
    [0.1, 0.9, 0.3, 0.7],
])
V_read = 0.2

def crossbar_compute(input_vec, G, V_read):
    V = np.array(input_vec) * V_read
    return G.T @ V, float(np.sum(V @ G))

print("Conductance matrix (secret weights):")
print(G)
I, P = crossbar_compute([1, 0, 1, 1], G, V_read)
print(f"Output currents: {np.round(I, 4)}")
print(f"Total power drawn: {P:.4f} W")
