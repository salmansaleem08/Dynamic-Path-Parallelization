#include <iostream>
#include <vector>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include <string>
#include <set>

#define ROOT 0

// Structure to represent a graph edge
struct Edge {
    int dest;
    int weight;
};

// Structure to represent edge modification (insertion or deletion)
struct EdgeModification {
    int src;
    int dest;
    int weight; // Only used for insertions
    bool is_insertion; // true for insertion, false for deletion
};

// Structure to represent a graph
struct Graph {
    int V; // Number of vertices
    std::vector<std::vector<Edge>> adj; // Adjacency list
};

// Create a hardcoded graph with 20 vertices and 50 edges
Graph createHardcodedGraph() {
    Graph graph;
    graph.V = 20;
    graph.adj.resize(graph.V);

    // Define edges: (source, destination, weight)
    std::vector<std::tuple<int, int, int>> edges = {
        {0, 1, 4}, {0, 2, 2}, {1, 3, 5}, {2, 1, 1}, {2, 3, 8},
        {3, 4, 3}, {4, 5, 6}, {5, 6, 2}, {6, 7, 7}, {7, 8, 4},
        {8, 9, 3}, {9, 10, 5}, {10, 11, 2}, {11, 12, 6}, {12, 13, 3},
        {13, 14, 4}, {14, 15, 5}, {15, 16, 2}, {16, 17, 3}, {17, 18, 4},
        {18, 19, 5}, {0, 3, 10}, {1, 4, 7}, {2, 5, 9}, {3, 6, 4},
        {4, 7, 3}, {5, 8, 6}, {6, 9, 2}, {7, 10, 5}, {8, 11, 4},
        {9, 12, 3}, {10, 13, 6}, {11, 14, 2}, {12, 15, 4}, {13, 16, 5},
        {14, 17, 3}, {15, 18, 2}, {16, 19, 6}, {0, 5, 12}, {1, 6, 8},
        {2, 7, 7}, {3, 8, 5}, {4, 9, 4}, {5, 10, 3}, {6, 11, 6},
        {7, 12, 2}, {8, 13, 5}, {9, 14, 4}, {10, 15, 3}, {11, 16, 2}
    };

    for (const auto& [u, v, w] : edges) {
        graph.adj[u].push_back({v, w});
    }

    std::cout << "Created graph with " << graph.V << " vertices and " << edges.size() << " edges" << std::endl;
    return graph;
}

// Partition the graph using METIS
void partitionGraph(const Graph& graph, int nparts, std::vector<int>& part) {
    idx_t nvtxs = graph.V;
    idx_t ncon = 1;

    idx_t total_edges = 0;
    for (int i = 0; i < nvtxs; i++) {
        total_edges += graph.adj[i].size();
    }

    idx_t* xadj = new idx_t[nvtxs + 1];
    idx_t* adjncy = new idx_t[total_edges];
    idx_t* adjwgt = new idx_t[total_edges];
    idx_t* vwgt = new idx_t[nvtxs];

    if (!xadj || !adjncy || !adjwgt || !vwgt) {
        std::cerr << "Memory allocation failed for METIS arrays" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (idx_t i = 0; i < nvtxs; ++i) {
        vwgt[i] = 1;
    }

    idx_t edge_count = 0;
    xadj[0] = 0;
    for (idx_t i = 0; i < nvtxs; ++i) {
        for (const auto& edge : graph.adj[i]) {
            if (edge_count < total_edges) {
                adjncy[edge_count] = edge.dest;
                adjwgt[edge_count] = edge.weight;
                edge_count++;
            }
        }
        xadj[i + 1] = edge_count;
    }

    part.resize(nvtxs);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    idx_t metis_nparts = nparts;
    idx_t objval = 0;

    std::cout << "Starting METIS partitioning with " << nparts << " parts..." << std::endl;

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, nullptr, adjwgt,
                                  &metis_nparts, nullptr, nullptr, options, &objval, part.data());

    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed with error code: " << ret << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> part_counts(nparts, 0);
    for (int p : part) {
        if (p >= 0 && p < nparts) {
            part_counts[p]++;
        } else {
            std::cerr << "Warning: Invalid partition value: " << p << std::endl;
        }
    }

    std::cout << "Partition sizes: ";
    for (int count : part_counts) {
        std::cout << count << " ";
    }
    std::cout << std::endl;

    delete[] xadj;
    delete[] adjncy;
    delete[] adjwgt;
    delete[] vwgt;
}

// Update SOSP tree for edge deletions
void updateSOSPDeletion(Graph& graph, std::vector<int>& dist, std::vector<int>& parent,
                       const std::vector<EdgeModification>& deletions, const std::vector<int>& part,
                       int rank, int size) {
    int n = graph.V;
    std::vector<std::vector<EdgeModification>> grouped_deletions(n); // Group by destination vertex
    std::vector<int> marked(n, 0); // Track affected vertices
    std::vector<int> affected;

    // Step 0: Preprocessing - Group deletions by destination vertex
    for (const auto& del : deletions) {
        if (!del.is_insertion) {
            grouped_deletions[del.dest].push_back(del);
        }
    }

    // Step 1: Process deleted edges and update graph
    #pragma omp parallel
    {
        std::vector<int> thread_affected;
        #pragma omp for nowait
        for (int v = 0; v < n; v++) {
            if (!part[v] == rank) continue; // Process only local vertices
            for (const auto& del : grouped_deletions[v]) {
                int u = del.src;
                // Remove edge (u, v) from graph
                auto& adj_list = graph.adj[u];
                for (auto it = adj_list.begin(); it != adj_list.end(); ++it) {
                    if (it->dest == v) {
                        adj_list.erase(it);
                        break;
                    }
                }
                // Check if vertex v's current shortest path used edge (u, v)
                if (parent[v] == u && dist[v] != std::numeric_limits<int>::max()) {
                    dist[v] = std::numeric_limits<int>::max();
                    parent[v] = -1;
                    thread_affected.push_back(v);
                    marked[v] = 1;
                }
            }
        }
        #pragma omp critical
        {
            affected.insert(affected.end(), thread_affected.begin(), thread_affected.end());
        }
    }

    // Step 2: Propagate updates
    bool global_changed = true;
    int iteration = 0;
    while (global_changed) {
        global_changed = false;
        iteration++;
        std::vector<int> next_affected;
        std::vector<int> neighbors;

        // Gather unique neighbors of affected vertices
        #pragma omp parallel
        {
            std::vector<int> thread_neighbors;
            #pragma omp for nowait
            for (size_t i = 0; i < affected.size(); i++) {
                int u = affected[i];
                if (!part[u] == rank) continue;
                for (const auto& edge : graph.adj[u]) {
                    thread_neighbors.push_back(edge.dest);
                }
            }
            #pragma omp critical
            {
                neighbors.insert(neighbors.end(), thread_neighbors.begin(), thread_neighbors.end());
            }
        }

        // Remove duplicates
        std::set<int> unique_neighbors(neighbors.begin(), neighbors.end());
        neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());

        // Update distances for neighbors
        #pragma omp parallel
        {
            std::vector<int> thread_affected;
            std::vector<std::tuple<int, int, int>> thread_updates; // (vertex, new_dist, parent)
            #pragma omp for nowait
            for (size_t i = 0; i < neighbors.size(); i++) {
                int v = neighbors[i];
                if (!part[v] == rank) continue;
                // Find alternative paths to v
                for (int u = 0; u < n; u++) {
                    for (const auto& edge : graph.adj[u]) {
                        if (edge.dest == v) {
                            int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                                          std::numeric_limits<int>::max() : dist[u] + edge.weight;
                            if (new_dist < dist[v]) {
                                thread_updates.emplace_back(v, new_dist, u);
                                if (marked[v] == 0) {
                                    thread_affected.push_back(v);
                                    marked[v] = 1;
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            {
                for (const auto& [v, new_dist, u] : thread_updates) {
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        parent[v] = u;
                    }
                }
                next_affected.insert(next_affected.end(), thread_affected.begin(), thread_affected.end());
            }
        }

        affected = next_affected;
        global_changed = !affected.empty();

        // Synchronize distances and parents
        std::vector<int> global_dist = dist;
        MPI_Allreduce(MPI_IN_PLACE, global_dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        std::vector<int> global_parent(n, -1);
        for (int i = 0; i < n; i++) {
            if (dist[i] != std::numeric_limits<int>::max() && parent[i] != -1) {
                global_parent[i] = parent[i];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, global_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Update local copies
        for (int i = 0; i < n; i++) {
            if (global_dist[i] < dist[i]) {
                dist[i] = global_dist[i];
                parent[i] = global_parent[i];
                if (part[i] == rank && marked[i] == 0) {
                    affected.push_back(i);
                    marked[i] = 1;
                    global_changed = true;
                }
            } else if (dist[i] == global_dist[i] && parent[i] != global_parent[i]) {
                parent[i] = global_parent[i];
            }
        }

        // Synchronize global_changed
        bool temp_changed = global_changed;
        MPI_Allreduce(&temp_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (rank == ROOT) {
            std::cout << "Deletion update iteration " << iteration << ", affected vertices: "
                      << affected.size() << ", changes: " << (global_changed ? "yes" : "no") << std::endl;
        }

        if (iteration > 100) {
            if (rank == ROOT) {
                std::cout << "Deletion update reached iteration limit" << std::endl;
            }
            break;
        }
    }

    // Final synchronization
    MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    std::vector<int> global_parent(n, -1);
    for (int i = 0; i < n; i++) {
        if (dist[i] != std::numeric_limits<int>::max() && parent[i] != -1) {
            global_parent[i] = parent[i];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, global_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    parent = global_parent;
}

// Parallel SOSP algorithm (modified to handle initial computation)
void parallelSOSP(Graph& graph, int source, std::vector<int>& dist,
                 std::vector<int>& parent, const std::vector<int>& part,
                 int rank, int size) {
    int n = graph.V;
    dist.assign(n, std::numeric_limits<int>::max());
    parent.assign(n, -1);

    if (source < 0 || source >= n) {
        if (rank == ROOT) {
            std::cerr << "Invalid source vertex: " << source << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    dist[source] = 0;
    parent[source] = source;

    std::vector<bool> is_local(n, false);
    int local_count = 0;
    for (int i = 0; i < n; ++i) {
        if (part[i] == rank) {
            is_local[i] = true;
            local_count++;
        }
    }
    std::cout << "Process " << rank << " owns " << local_count << " vertices" << std::endl;

    std::vector<int> active_vertices;
    if (is_local[source]) {
        active_vertices.push_back(source);
        std::cout << "Process " << rank << " owns source vertex " << source << std::endl;
    }

    bool global_changed = true;
    int iteration = 0;

    while (global_changed) {
        bool local_changed = false;
        iteration++;

        std::vector<int> next_active;

        #pragma omp parallel
        {
            std::vector<int> thread_active;
            std::vector<std::tuple<int, int, int>> thread_updates; // (vertex, new_dist, parent)

            #pragma omp for nowait
            for (size_t i = 0; i < active_vertices.size(); i++) {
                int u = active_vertices[i];
                if (!is_local[u]) continue;

                for (const Edge& edge : graph.adj[u]) {
                    int v = edge.dest;
                    int new_dist = dist[u] + edge.weight;

                    if (new_dist < dist[v]) {
                        thread_updates.emplace_back(v, new_dist, u);
                        thread_active.push_back(v);
                    }
                }
            }

            #pragma omp critical
            {
                for (const auto& [v, new_dist, u] : thread_updates) {
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        parent[v] = u;
                        local_changed = true;
                        next_active.push_back(v);
                    }
                }
            }

            #pragma omp critical
            {
                thread_active.swap(next_active);
            }
        }

        active_vertices = next_active;

        // Synchronize distances across processes
        std::vector<int> global_dist = dist;
        MPI_Allreduce(MPI_IN_PLACE, global_dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Update distances and mark vertices for next iteration
        for (int i = 0; i < n; i++) {
            if (global_dist[i] < dist[i]) {
                dist[i] = global_dist[i];
                local_changed = true;
                if (is_local[i]) {
                    active_vertices.push_back(i);
                }
            }
        }

        // Synchronize parent array
        std::vector<int> global_parent(n, -1);
        for (int i = 0; i < n; i++) {
            if (dist[i] != std::numeric_limits<int>::max() && parent[i] != -1) {
                global_parent[i] = parent[i];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, global_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        parent = global_parent;

        // Check for global changes
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (rank == ROOT) {
            std::cout << "Iteration " << iteration << " completed, active vertices: "
                      << active_vertices.size() << ", changes: "
                      << (global_changed ? "yes" : "no") << std::endl;
        }

        if (iteration > 100) {
            std::cout << "Process " << rank << " reached iteration limit" << std::endl;
            break;
        }
    }

    // Final synchronization
    MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    std::vector<int> global_parent(n, -1);
    for (int i = 0; i < n; i++) {
        if (dist[i] != std::numeric_limits<int>::max() && parent[i] != -1) {
            global_parent[i] = parent[i];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, global_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    parent = global_parent;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int source = 0; // Default source vertex
    if (argc > 1) {
        source = std::stoi(argv[1]);
    }

    if (rank == ROOT) {
        std::cout << "Starting SOSP computation with " << size << " MPI processes" << std::endl;
        std::cout << "Source vertex: " << source << std::endl;
    }

    // Create graph
    Graph graph;
    if (rank == ROOT) {
        std::cout << "Creating hardcoded graph..." << std::endl;
        graph = createHardcodedGraph();
    }

    // Broadcast number of vertices
    int V;
    if (rank == ROOT) {
        V = graph.V;
    }
    MPI_Bcast(&V, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank != ROOT) {
        graph.V = V;
        graph.adj.resize(V);
    }

    // Broadcast graph data
    for (int u = 0; u < V; ++u) {
        int adj_size;
        if (rank == ROOT) {
            adj_size = graph.adj[u].size();
        }
        MPI_Bcast(&adj_size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if (rank != ROOT) {
            graph.adj[u].resize(adj_size);
        }

        if (adj_size > 0) {
            std::vector<int> edge_data;
            if (rank == ROOT) {
                edge_data.resize(adj_size * 2);
                for (int i = 0; i < adj_size; i++) {
                    edge_data[i*2] = graph.adj[u][i].dest;
                    edge_data[i*2+1] = graph.adj[u][i].weight;
                }
            } else {
                edge_data.resize(adj_size * 2);
            }

            MPI_Bcast(edge_data.data(), adj_size * 2, MPI_INT, ROOT, MPI_COMM_WORLD);

            if (rank != ROOT) {
                for (int i = 0; i < adj_size; i++) {
                    graph.adj[u][i].dest = edge_data[i*2];
                    graph.adj[u][i].weight = edge_data[i*2+1];
                }
            }
        }
    }

    if (rank == ROOT) {
        std::cout << "Finished broadcasting graph data to all processes" << std::endl;
    }

    // Partition the graph
    std::vector<int> part(V);
    if (rank == ROOT) {
        std::cout << "Partitioning graph..." << std::endl;
        partitionGraph(graph, size, part);
    }
    MPI_Bcast(part.data(), V, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        std::cout << "Finished graph partitioning" << std::endl;
    }

    // Compute initial shortest paths
    std::vector<int> dist, parent;
    double start_time = MPI_Wtime();

    if (rank == ROOT) {
        std::cout << "Starting initial SOSP computation..." << std::endl;
    }

    parallelSOSP(graph, source, dist, parent, part, rank, size);
    double end_time = MPI_Wtime();

    if (rank == ROOT) {
        std::cout << "Initial SOSP computation completed in " << (end_time - start_time) << " seconds" << std::endl;
    }

    // Define sample edge deletions
    std::vector<EdgeModification> deletions;
    if (rank == ROOT) {
        deletions = {
            {0, 1, 0, false}, // Delete edge (0,1)
            {9, 14, 0, false}  // Delete edge (2,3)
        };
        std::cout << "Applying edge deletions..." << std::endl;
    }

    // Broadcast deletions
    int num_deletions;
    if (rank == ROOT) {
        num_deletions = deletions.size();
    }
    MPI_Bcast(&num_deletions, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank != ROOT) {
        deletions.resize(num_deletions);
    }

    struct MPIData {
        int src;
        int dest;
        int weight;
        int is_insertion;
    };

    std::vector<MPIData> mpi_deletions(num_deletions);
    if (rank == ROOT) {
        for (size_t i = 0; i < deletions.size(); i++) {
            mpi_deletions[i] = {deletions[i].src, deletions[i].dest, deletions[i].weight, deletions[i].is_insertion ? 1 : 0};
        }
    }

    MPI_Bcast(mpi_deletions.data(), num_deletions * sizeof(MPIData) / sizeof(int), MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank != ROOT) {
        for (int i = 0; i < num_deletions; i++) {
            deletions[i] = {mpi_deletions[i].src, mpi_deletions[i].dest, mpi_deletions[i].weight, mpi_deletions[i].is_insertion == 1};
        }
    }

    // Update SOSP for deletions
    start_time = MPI_Wtime();
    if (rank == ROOT) {
        std::cout << "Starting SOSP update for edge deletions..." << std::endl;
    }

    updateSOSPDeletion(graph, dist, parent, deletions, part, rank, size);
    end_time = MPI_Wtime();

    if (rank == ROOT) {
        std::cout << "SOSP update for deletions completed in " << (end_time - start_time) << " seconds" << std::endl;
    }

    // Print results on root
    if (rank == ROOT) {
        std::cout << "\nFinal results after edge deletions:" << std::endl;
        std::cout << "Total execution time for update: " << (end_time - start_time) << " seconds" << std::endl;

        int sample_count = std::min(20, V);
        std::cout << "\nShortest paths (first " << sample_count << " vertices):" << std::endl;
        std::cout << "Vertex\tDistance\tPath" << std::endl;
        for (int i = 0; i < sample_count; ++i) {
            if (dist[i] != std::numeric_limits<int>::max()) {
                std::cout << i << "\t" << dist[i] << "\t\t";
                std::vector<int> path;
                for (int v = i; v != source && v != -1; v = parent[v]) {
                    path.push_back(v);
                    if (path.size() > V) {
                        std::cout << "[loop detected]";
                        break;
                    }
                }
                path.push_back(source);
                for (auto it = path.rbegin(); it != path.rend(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << i << "\tINF\t\tNo path" << std::endl;
            }
        }

        int reachable = 0;
        for (int d : dist) {
            if (d != std::numeric_limits<int>::max()) {
                reachable++;
            }
        }
        std::cout << "\nReachable vertices: " << reachable << "/" << V << std::endl;
    }

    MPI_Finalize();
    return 0;
}
