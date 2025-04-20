# Parallel SOSP and MOSP Update Algorithms

## Project Description

This repository is dedicated to the study and future implementation of parallel algorithms for updating Single-Objective Shortest Paths (SOSP) and Multi-Objective Shortest Paths (MOSP) in large dynamic networks, inspired by the research paper:

> **"Parallel Algorithm for Updating Multi-objective Shortest Paths in Large Dynamic Networks"**  
> *Arindam Khanda, S. M. Shovan, and Sajal K. Das (SC-W 2023, Denver, CO)*

The project explores efficient parallel solutions for path updates in networks with frequent edge insertions — applicable to transportation, social, communication, and drone delivery systems.

---

## Paper Overview

The paper proposes parallel algorithms to update SOSP and MOSP in large, dynamic graphs. Key contributions include:

- **Focus**: Scalable parallel SOSP updates and heuristic-based MOSP updates using shared-memory systems.
- **Problem**: Frequent edge insertions in dynamic graphs require efficient updates to avoid costly full recomputation.
- **Applications**: Road and drone networks, social media graphs, and communication infrastructure.

---

## Key Concepts

### Dynamic Networks
- Directed graphs with frequent edge changes.
- Requires real-time path update mechanisms.

### Single-Objective Shortest Path (SOSP)
- Computes shortest path tree based on one objective (e.g., distance).
- Efficiently updated after edge insertions.

### Multi-Objective Shortest Path (MOSP)
- Considers multiple objectives (e.g., time, fuel).
- Produces Pareto-optimal paths.
- Uses dominance filtering to eliminate non-optimal routes.

---

## Algorithms

### Algorithm 1: SOSP Update
**Goal**: Update SOSP tree after edge insertions.

**Steps**:
1. **Preprocessing**: Group inserted edges by endpoint.
2. **Edge Processing**: Parallel updates for affected vertices.
3. **Propagation**: Iteratively update neighbors until stable.

**Features**:
- Grouping prevents race conditions.
- Highly parallelizable with OpenMP.

---

### Algorithm 2: MOSP Update
**Goal**: Heuristically update a single Pareto-optimal MOSP.

**Steps**:
1. Refresh SOSP trees using Algorithm 1.
2. Merge SOSP trees into a combined graph.
3. Compute shortest path on the merged graph.
4. Reassign original weights for MOSP interpretation.

**Features**:
- Converts complex MOSP into efficient SOSP.
- Allows objective prioritization.

---

## Theoretical Foundations

- **Theorem 1**: Combined graphs guarantee Pareto-optimality.
- **Lemma 2**: SOSP paths form valid subpaths of MOSPs.
- **Theorem 3**: Ensures optimal MOSP updates in dynamic graphs.

---

## Implementation Overview

- **Language**: C++  
- **Parallelization**: OpenMP (shared memory)  
- **Test System**: Dual 32-core CPUs, 64 GB RAM

### Data Structures
- **Graph**: Adjacency list
- **SOSP Tree**: Parent-child structure
- **Edge Tracker**: Tracks modified edges
- **Affected Vertices**: Flags updates

### Parallelization Strategy
- **SOSP**: Parallel edge and neighbor updates
- **MOSP**: Sequential SOSP updates, parallel combined graph creation

---

## Performance Evaluation

### Datasets
- Large-scale road networks
- Random geometric graphs

### Observations
- Thread count improves speed.
- Sparse graphs scale better.
- SOSP updates dominate execution time.
- No existing parallel MOSP algorithms for dynamic graphs.

---

## Key Contributions

- **Parallel SOSP Update**: Avoids race conditions with grouping.
- **Heuristic MOSP Update**: Reduces complexity to SOSP form.
- **Performance**: Up to 15× speedup in real and synthetic datasets.
- **Theory**: Proven Pareto-optimality.
- **Use Cases**: Drone delivery, traffic networks, and more.

---

## Related Work

- **MOSP**: Previously limited to bi-objective models.
- **Parallel SOSP**: Existing work focuses on single-objective cases.
- **Dynamic MOSP**: Lacks parallel updates in prior literature.

---

## Limitations

- Only supports **edge insertions** (not deletions).
- MOSP's **SOSP updates are sequential**.
- **Shared-memory** model limits scalability to larger systems.

---

## Future Work

- Support edge deletions.
- Hybrid parallel SOSP tree updates.
- Distributed-memory implementation using MPI.
- GPU-accelerated versions for dense graphs.

---

## Conclusion

This project builds on pioneering research in dynamic pathfinding, introducing scalable, efficient, and provably correct algorithms for parallel SOSP and heuristic MOSP updates. The future implementation will explore MPI, OpenMP, and METIS to extend scalability to massive networks.

---

## References

Khanda, A., Shovan, S. M., & Das, S. K. (2023).  
*Parallel Algorithm for Updating Multi-objective Shortest Paths in Large Dynamic Networks*.  
SC-W 2023, Denver, CO.


## Contributors

- **Muhammad Salman Saleem 22I-0904 G**
- **Muneeb Amir 22I-1188 G**
- **Zuhaak Ahmad 22I-1352 G**  


---