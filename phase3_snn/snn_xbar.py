"""
Phase 3 — SNN + memristor crossbar co-design
Run standalone: python phase3_snn/snn_xbar.py
Key concept: trained SNN weights mapped to hardware conductance values.
             Device noise degrades accuracy — shows why security matters.
"""
import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

# Simplified forward pass with a pretrained weight set
W1 = np.array([[ 4.1, -3.8,  3.9, -4.2],
               [-4.0,  3.7, -3.8,  4.1]])
b1 = np.array([ 1.2, -1.1,  1.3, -1.0])
W2 = np.array([[ 6.0], [-5.8], [ 5.9], [-6.1]])
b2 = np.array([-2.8])

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

# Ideal run
a1 = np.tanh(X @ W1 + b1)
pred = (sigmoid(a1 @ W2 + b2).flatten() > 0.5).astype(int)
print(f"Ideal accuracy: {np.mean(pred == y)*100:.0f}%")

# Noisy crossbar run (simulate device variability)
G_min, G_max = 0.01, 1.0
W1_norm = (W1 - W1.min()) / (W1.max() - W1.min() + 1e-9)
G_mapped = G_min + W1_norm * (G_max - G_min)
G_noisy  = G_mapped + np.random.normal(0, 0.15, G_mapped.shape)
G_noisy  = np.clip(G_noisy, G_min, G_max)
W1_approx = (G_noisy - G_min) / (G_max - G_min) * (W1.max() - W1.min()) + W1.min()
a1n = np.tanh(X @ W1_approx + b1)
pred_n = (sigmoid(a1n @ W2 + b2).flatten() > 0.5).astype(int)
print(f"Noisy crossbar accuracy (σ=0.15): {np.mean(pred_n == y)*100:.0f}%")
