"""
Phase 4 — Memristive PUF defense
Run standalone: python phase4_defense/puf.py
Key concept: device variability → unique chip fingerprint = security primitive.
Hamming distance ≈ 50%  means chips are maximally distinguishable.
"""
import numpy as np

def make_puf(seed, n_challenges=64, n_bits=8, sigma=0.08):
    np.random.seed(seed)
    variation = np.random.normal(1.0, sigma, n_bits)
    responses = []
    for c in range(n_challenges):
        challenge = np.array(list(map(int, f"{c:08b}")), dtype=float)
        responses.append(int((challenge * variation).sum() > n_bits * 0.5))
    return np.array(responses)

chip_A = make_puf(seed=1)
chip_B = make_puf(seed=2)
hd = np.mean(chip_A != chip_B) * 100
print(f"Chip A responses: {chip_A[:16]} ...")
print(f"Chip B responses: {chip_B[:16]} ...")
print(f"Inter-chip Hamming distance: {hd:.1f}%  (ideal ≈ 50%)")
