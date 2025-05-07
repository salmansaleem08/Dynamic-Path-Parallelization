# Parallel SOSP and MOSP Update Algorithms

## Project Description

This repository implements parallel algorithms for updating Single-Objective Shortest Paths (SOSP) and Multi-Objective Shortest Paths (MOSP) in large dynamic networks, based on the research paper:

**"Parallel Algorithm for Updating Multi-objective Shortest Paths in Large Dynamic Networks"**  
*Arindam Khanda, S. M. Shovan, and Sajal K. Das (SC-W 2023, Denver, CO)*

The project provides efficient parallel solutions for path updates in dynamic networks with frequent edge insertions and deletions, targeting applications in transportation, drone delivery, social networks, and communication systems.

---

## Paper Overview

The paper presents scalable parallel algorithms for updating SOSP and MOSP in dynamic graphs. Key points include:

- **Objective**: Efficient parallel updates for SOSP and heuristic-based MOSP in shared-memory and distributed systems.  
- **Problem**: Frequent edge changes (insertions/deletions) require updates without full recomputation.  
- **Applications**: Real-time pathfinding in road networks, drone navigation, and communication infrastructures.

---

## Key Concepts

### Dynamic Networks
- Directed graphs with frequent edge modifications.
- Require real-time path update mechanisms.

### Single-Objective Shortest Path (SOSP)
- Computes shortest path tree for a single objective (e.g., distance).
- Supports efficient updates after edge changes.

### Multi-Objective Shortest Path (MOSP)
- Considers multiple objectives (e.g., time, fuel).
- Produces Pareto-optimal paths via dominance filtering.

---

## Algorithms

### Algorithm 1: Parallel SOSP Update

**Goal**: Update SOSP tree after edge insertions/deletions.  
**Steps**:
1. **Preprocessing**: Group edges by destination to prevent race conditions.  
2. **Edge Processing**: Parallel updates for affected vertices.  
3. **Propagation**: Iteratively update neighbors until convergence.  

**Features**:
- Race-free edge grouping.
- Parallelized with OpenMP (shared-memory) and MPI (distributed-memory).
- Uses METIS for graph partitioning.

### Algorithm 2: Heuristic MOSP Update

**Goal**: Update a single Pareto-optimal MOSP using SOSP solutions.  
**Steps**:
1. Update SOSP trees for each objective using Algorithm 1.  
2. Merge SOSP trees into a combined graph.  
3. Compute shortest path on the combined graph using Bellman-Ford.  
4. Reassign original weights for MOSP.  

**Features**:
- Simplifies MOSP to SOSP computations.
- Ensures Pareto-optimality with theoretical guarantees.
- Supports objective prioritization.

---

## Theoretical Foundations

- **Theorem 1**: Combined graph ensures Pareto-optimality.
- **Lemma 2**: SOSP paths are valid subpaths of MOSPs.
- **Theorem 3**: Guarantees correct MOSP updates in dynamic graphs.

---

## Implementation Overview

- **Language**: C++
- **Parallelization**:
  - MPI: Distributed-memory parallelism.
  - OpenMP: Shared-memory multi-threading.
  - METIS: Graph partitioning for load balancing.

### Files
- `sosp_update_parallel_openmp_mpi.cpp`: Hybrid OpenMP + MPI implementation.
- `sosp_update_parallel_mpi.cpp`: MPI-only implementation.
- `serial_mosp.cpp`: Serial baseline implementation.

- **Input**: `weighted_graph_usa.txt` (format: source, destination, weight1, [weight2]).
- **Test System**: Dual 32-core CPUs, 64 GB RAM.

---

## Data Structures

- **Graph**: Adjacency list.
- **SOSP Tree**: Parent-child structure with distances.
- **Edge Tracker**: Tracks edge modifications.
- **Affected Vertices**: Flags vertices for updates.

---

## Parallelization Strategy

- **SOSP**: Parallel edge and neighbor updates.
- **MOSP**: Sequential SOSP updates, parallel combined graph creation.
- **Partitioning**: METIS balances workload across MPI processes.

---

## Running Commands

### Prerequisites

- **Compiler**: `mpicxx` (MPI-enabled), `g++` (serial).
- **Libraries**: MPI, OpenMP, METIS (`libmetis`).
- **Input File**: `weighted_graph_usa.txt` in the working directory.
- **Hostfile**: `hosts.txt` specifying compute nodes (e.g., `node1 slots=4`).

---

### 1. MPI-Only Implementation

**File**: `Parallel_Algorithm_Using_MPI_BigData.cpp`

**Compile**:
```bash
mpicxx -o parallel_mosp Parallel_Algorithm_Using_MPI_BigData.cpp -lmetis -I/usr/include -L/usr/lib
```

**Run**:
```bash
mpirun --hostfile hosts.txt -np 4 --mca plm_rsh_no_tree_spawn 1 ./parallel_mosp 2
```

- `-np 4`: 4 MPI processes  
- `2`: 2 objectives (e.g., time, fuel)

---

### 2. OpenMP + MPI Implementation

**File**: `Parallel_Algorithm_Using_OPENMP_MPI_BigData.cpp`

**Compile**:
```bash
mpicxx -o parallel_mosp Parallel_Algorithm_Using_OPENMP_MPI_BigData.cpp -fopenmp -lmetis -I/usr/include -L/usr/lib
```

**Run**:
```bash
mpirun --hostfile hosts.txt -np 4 --mca plm_rsh_no_tree_spawn 1 ./parallel_mosp 2
```

---

### 3. Serial Implementation

**File**: `Parallel_Algorithm_Using_SerialApproach_Only.cpp`

**Compile**:
```bash
g++ -o serial_mosp MOSP_Using_Serial_BigData.cpp -lmetis -I/usr/include -L/usr/lib
```

**Run**:
```bash
./serial_mosp 2
```

---

### Notes

- Ensure `weighted_graph_usa.txt` is present.
- Adjust `-np` based on available nodes/cores.
- Update METIS library paths (`-I/usr/include`, `-L/usr/lib`) if needed.

---

## Performance Evaluation

### Datasets

- **Real-World**: USA road networks (`weighted_graph_usa.txt`).
- **Synthetic**: Random geometric graphs.

### Observations

- **Speedup**: Up to 15× faster than serial implementation.
- **Scalability**: Better performance with more threads/processes, especially on sparse graphs.
- **Bottlenecks**: SOSP updates dominate MOSP computation time.
- **Novelty**: First parallel MOSP update algorithm for dynamic graphs.

---

## Key Contributions

- **Parallel SOSP Update**: Race-free edge grouping for efficient updates.
- **Heuristic MOSP Update**: Reduces MOSP complexity to SOSP.
- **Performance**: Significant speedups on real/synthetic datasets.
- **Theory**: Proven Pareto-optimality and correctness.
- **Applications**: Real-time pathfinding for dynamic systems.

---

## Related Work

- **MOSP**: Limited to bi-objective or static graphs in prior work.
- **Parallel SOSP**: Focused on single-objective updates.
- **Dynamic MOSP**: No prior parallel solutions for edge changes.

---

## Limitations

- Supports only edge insertions/deletions.
- Sequential SOSP updates for MOSP across objectives.
- Scalability constrained by shared-memory (OpenMP) or distributed-memory (MPI) models.

---

## Future Work

- Support edge weight changes and vertex modifications.
- Fully parallel SOSP updates for MOSP.
- Distributed MOSP updates using MPI.
- GPU-accelerated implementations for dense graphs.

---

## Conclusion

This project delivers scalable parallel algorithms for SOSP and MOSP updates in dynamic networks. The implementations (`sosp_update_parallel_openmp_mpi.cpp`, `sosp_update_parallel_mpi.cpp`, `serial_mosp.cpp`) offer significant performance improvements, with applications in real-time pathfinding. Future work will enhance scalability and support broader graph modifications.

---

## References

Khanda, A., Shovan, S. M., & Das, S. K. (2023). *Parallel Algorithm for Updating Multi-objective Shortest Paths in Large Dynamic Networks.* SC-W 2023, Denver, CO.

---

## Contributors

- Muhammad Salman Saleem — 22I-0904 G  
- Muneeb Amir — 22I-1188 G  
- Zuhaak Ahmad — 22I-1352 G  
