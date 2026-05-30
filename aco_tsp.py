
import numpy as np
import random

# ─────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────
NODE_NAMES = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']      # daftar nama node
IDX = {n: i for i, n in enumerate(NODE_NAMES)}                  # mapping nama node ke index matrix
N = len(NODE_NAMES)                                             # jmlh node
INF = 9999                                                      # mnyatakan tidak ada edge / representasi jarak tak terhingga

d = np.full((N, N), INF, dtype=float)       # matriks jarak antar node, diisi dengan INF (tidak ada edge) secara default
np.fill_diagonal(d, 0)                      # jarak dari node ke dirinya sendiri = 0  

# bobot jalan antara dua node, dari segala arah (krn undirected graph)
def edge(u, v, w):
    d[IDX[u]][IDX[v]] = w       # U ke V
    d[IDX[v]][IDX[u]] = w       # V ke U 

# edge yang ada di graph dan bobotnya (13 edges)
edge('#', 'A', 3);  edge('#', 'C', 2);  edge('#', 'G', 5)
edge('A', 'C', 6);  edge('C', 'B', 9);  edge('C', 'F', 4)
edge('B', 'D', 8);  edge('D', 'E', 7);  edge('D', 'H', 9)
edge('E', 'F', 2);  edge('E', 'G', 1);  edge('E', 'H', 1)
edge('G', 'H', 3)

# Define constants for start, end, and checkpoint nodes
START = IDX['H']
END   = IDX['D']
CHECK = IDX['#']


# ─────────────────────────────────────────────────────────
# ACO PARAMETERS
# ─────────────────────────────────────────────────────────
N_ANTS = 5     # jmlh semut tiap iteration
N_ITER = 10    # jmlh iterasi ACO
ALPHA  = 0.7    # bobot pengaruh pheromone
BETA   = 4.18    # bobot pengaruh heuristik (jarak/visibility)
RHO    = 0.5    # evaporation rate pheromone
Q      = 100.0  # konstanta deposit pheromone

pheromone = 0.1 * np.ones((N, N))       # matrix level pheromone awal, diisi dengan nilai kecil (0.1) untuk semua edge

# Visibility = 1/distance for direct edges only
visibility = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j and d[i][j] < INF:
            visibility[i][j] = 1.0 / d[i][j]        # visibility hanya untuk edge yang ada, 0 untuk yang tidak ada

# ─────────────────────────────────────────────────────────
# WALK: probabilistic simple path from src to dst
# ─────────────────────────────────────────────────────────
def walk_leg(src, dst, forbidden):
    path    = [src]
    visited = set(forbidden) | {src}        # blokir node spy gak dilewati lagi 
    current = src

    while current != dst:
        candidates = [j for j in range(N)                               
                      if d[current][j] < INF and j not in visited]          # pilih calon node tetangga yang terhubung, dan blm dikunjingin
        if not candidates:
            return None

        scores = [pheromone[current][j] ** ALPHA * visibility[current][j] ** BETA       
                  for j in candidates]                                                 # hitung skor untuk setiap kandidat berdasarkan pheromone dan visibility
        total  = sum(scores)
        probs  = [s / total for s in scores] if total > 0 \
                 else [1.0 / len(candidates)] * len(candidates)     # normalisasi skor jadi probabilitas, jika total 0 (semua skor 0), bagi rata

        r = random.random(); cumul = 0.0; chosen = candidates[-1]
        for j, p in zip(candidates, probs):
            cumul += p
            if r <= cumul:
                chosen = j
                break

        path.append(chosen)
        visited.add(chosen)
        current = chosen

    return path

# spaya lewat checkpoint #
def build_path():
    """Full simple path H -> # -> D."""
    # H ke #
    leg1 = walk_leg(START, CHECK, forbidden=set())
    if leg1 is None:
        return None

    # checkpoint # ke D, gak pakai node di leg1 (kec. #)
    leg2 = walk_leg(CHECK, END, forbidden=set(leg1[:-1]))
    if leg2 is None:
        return None

    # gabung leg1 dan leg2
    full = leg1 + leg2[1:]
    if len(full) != len(set(full)):
        return None

    cost = sum(d[full[i]][full[i+1]] for i in range(len(full) - 1))     # biaya total
    return (full, cost) if cost < INF else None                         


# ─────────────────────────────────────────────────────────
# ACO MAIN LOOP
# ─────────────────────────────────────────────────────────
def run_aco():
    global pheromone
    best_path = None
    best_cost = float('inf')

    print("=" * 60)
    print("  ACO — Shortest Path  H -> # -> D  (no node reuse)")
    print(f"  Ants={N_ANTS}, Iter={N_ITER}, α={ALPHA}, β={BETA}, ρ={RHO}")
    print("=" * 60)
    print(f"\n{'Iter':>5}  {'Best Cost':>10}  Path")
    print("-" * 60)

    for it in range(N_ITER):
        iter_paths, iter_costs = [], []

        for _ in range(N_ANTS):
            result = build_path()
            if result is not None:
                p, c = result
                iter_paths.append(p)
                iter_costs.append(c)
                if c < best_cost:
                    best_cost = c
                    best_path = p[:]

        # Evaporation
        pheromone *= (1 - RHO)                           # jejak berkurang agar tidak menumpuk terus
        np.clip(pheromone, 1e-6, None, out=pheromone)

        # Deposit — elitist: iteration best gets 3x deposit
        iter_best_count = 0
        if iter_paths:
            min_c = min(iter_costs)
            iter_best_count = iter_costs.count(min_c)
            for path, cost in zip(iter_paths, iter_costs):
                deposit = Q / cost              # untuk tiap path valid, deposit pheromone proporsional dengan kualitasnya (Q / cost), jalur pendek lebih banyak pheromone
                if cost == min_c:               
                    deposit *= 3                # iteration best dapat bonus deposit 3x
                for k in range(len(path) - 1):
                    a, b = path[k], path[k+1]
                    pheromone[a][b] += deposit
                    pheromone[b][a] += deposit

        exact_best_count = 0
        if best_path is not None:
            exact_best_count = sum(1 for p in iter_paths if p == best_path)

        # print progress tiap iterasi
        if best_path:
            pstr = " -> ".join(NODE_NAMES[n] for n in best_path)
            print(f"{it+1:5d}  {best_cost:10.1f}  valid={len(iter_paths):2d}  best={iter_best_count:2d}  same={exact_best_count:2d}  {pstr}")
        else:
            print(f"{it+1:5d}  {'no valid path yet':>16}")

    return best_path, best_cost


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)         # untuk reproducibility, set seed random agar hasilnya sama setiap kali dijalankan
    np.random.seed(42)

    # print graph edge dan bobotnya
    print("\nGraph edges:")
    for i in range(N):
        for j in range(i+1, N):
            if d[i][j] < INF:
                print(f"  {NODE_NAMES[i]} - {NODE_NAMES[j]} = {int(d[i][j])}")
    print()

    # menjalankan ACO untuk mencari jalur terbaik
    best_path, best_cost = run_aco()

    # cetak hasil final
    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)
    if best_path:
        names = [NODE_NAMES[n] for n in best_path]
        print(f"  Best path : {' -> '.join(names)}")
        print(f"  Cost      : {int(best_cost)}")
        print(f"\n  Edge breakdown:")
        total = 0
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i+1]
            w = int(d[u][v])
            total += w
            print(f"    {NODE_NAMES[u]} -> {NODE_NAMES[v]} : {w}")
        print(f"    Total = {total}")
        print(f"\n  # checkpoint at position {names.index('#')} ")
        print(f"  No node reused: {len(names) == len(set(names))} ")
    else:
        print("  No valid path found.")
