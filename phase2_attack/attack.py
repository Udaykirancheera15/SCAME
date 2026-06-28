"""
Phase 2 — Power side-channel attack + fault injection
Run standalone: python phase2_attack/attack.py
Key concept: if power varies with input, secret weights can be inferred.
"""
import numpy as np

G = np.array([
    [0.8, 0.2, 0.9, 0.1],
    [0.3, 0.7, 0.4, 0.6],
    [0.5, 0.5, 0.8, 0.2],
    [0.1, 0.9, 0.3, 0.7],
])
V_read = 0.2

all_inputs = [list(map(int, f"{i:04b}")) for i in range(16)]
traces = [float(np.sum(np.array(inp) * V_read @ G)) for inp in all_inputs]
print("Power trace for each input:")
for i, (inp, p) in enumerate(zip(all_inputs, traces)):
    print(f"  Input {inp}  →  power = {p:.4f} A")
print(f"\nRange: {max(traces) - min(traces):.4f} A  ← the side-channel leak")
