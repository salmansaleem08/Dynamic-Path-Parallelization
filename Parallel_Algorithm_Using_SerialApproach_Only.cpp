#include <iostream>
#include <vector>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <functional>

// Structure to represent a graph edge
struct Edge {
    int dest;
    std::vector<int> weights; // Weight vector for multiple objectives
};

// Structure to represent edge modification (insertion or deletion)
struct EdgeModification {
    int src;
    int dest;
    std::vector<int> weights; // Weight vector for insertions
    bool is_insertion; // true for insertion, false for deletion
};

// Structure to represent a graph
struct Graph {
    int V; // Number of vertices
    std::vector<std::vector<Edge>> adj; // Adjacency list
};

// Structure to represent an SOSP tree
struct SOSPTree {
    std::vector<std::pair<int, int>> edges; // Edges in the tree (u, v)
    std::vector<int> dist; // Distances from source
    std::vector<int> parent; // Parent array
};

// Create a hardcoded graph with 20 vertices and 50 edges, supporting multiple objectives
Graph createHardcodedGraph(int num_objectives) {
    Graph graph;
    graph.V = 20;
    graph.adj.resize(graph.V);

    // Define edges: (source, destination, weight_vector)
    std::vector<std::tuple<int, int, std::vector<int>>> edges = {
        {0, 1, {4, 5}}, {0, 2, {2, 3}}, {1, 3, {5, 4}}, {2, 1, {1, 2}}, {2, 3, {8, 7}},
        {3, 4, {3, 3}}, {4, 5, {6, 5}}, {5, 6, {2, 4}}, {6, 7, {7, 6}}, {7, 8, {4, 5}},
        {8, 9, {3, 3}}, {9, 10, {5, 4}}, {10, 11, {2, 2}}, {11, 12, {6, 5}}, {12, 13, {3, 4}},
        {13, 14, {4, 3}}, {14, 15, {5, 5}}, {15, 16, {2, 2}}, {16, 17, {3, 3}}, {17, 18, {4, 4}},
        {18, 19, {5, 5}}, {0, 3, {10, 9}}, {1, 4, {7, 6}}, {2, 5, {9, 8}}, {3, 6, {4, 4}},
        {4, 7, {3, 3}}, {5, 8, {6, 5}}, {6, 9, {2, 2}}, {7, 10, {5, 4}}, {8, 11, {4, 3}},
        {9, 12, {3, 3}}, {10, 13, {6, 5}}, {11, 14, {2, 2}}, {12, 15, {4, 4}}, {13, 16, {5, 5}},
        {14, 17, {3, 3}}, {15, 18, {2, 2}}, {16, 19, {6, 6}}, {0, 5, {12, 11}}, {1, 6, {8, 7}},
        {2, 7, {7, 6}}, {3, 8, {5, 5}}, {4, 9, {4, 4}}, {5, 10, {3, 3}}, {6, 11, {6, 5}},
        {7, 12, {2, 2}}, {8, 13, {5, 4}}, {9, 14, {4, 3}}, {10, 15, {3, 3}}, {11, 16, {2, 2}}
    };

    // Ensure num_objectives is at least 1 to avoid empty weight vectors
    int effective_objectives = std::max(1, num_objectives);

    for (const auto& [u, v, w] : edges) {
        Edge edge;
        edge.dest = v;
        edge.weights = w;
        if (edge.weights.size() < static_cast<size_t>(effective_objectives)) {
            edge.weights.resize(effective_objectives, w.empty() ? 1 : w[0]); // Pad with first weight or 1
        } else if (edge.weights.size() > static_cast<size_t>(effective_objectives)) {
            edge.weights.resize(effective_objectives); // Truncate to effective_objectives
        }
        graph.adj[u].push_back(edge);
    }

    std::cout << "Created graph with " << graph.V << " vertices, " << edges.size()
              << " edges, and " << effective_objectives << " objectives" << std::endl;
    return graph;
}

// Serial Dijkstra's algorithm for SOSP
void serialSOSP(Graph& graph, int source, int objective_idx, std::vector<int>& dist,
                std::vector<int>& parent) {
    int n = graph.V;
    dist.assign(n, std::numeric_limits<int>::max());
    parent.assign(n, -1);

    if (source < 0 || source >= n) {
        std::cerr << "Invalid source vertex: " << source << std::endl;
        exit(1);
    }

    dist[source] = 0;
    parent[source] = source;

    std::set<std::pair<int, int>> pq; // {distance, vertex}
    pq.emplace(0, source);

    while (!pq.empty()) {
        int u = pq.begin()->second;
        int d = pq.begin()->first;
        pq.erase(pq.begin());

        if (d > dist[u]) continue;

        for (const Edge& edge : graph.adj[u]) {
            int v = edge.dest;
            int weight = edge.weights.empty() ? 1 : edge.weights[objective_idx % edge.weights.size()];
            int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                           std::numeric_limits<int>::max() : dist[u] + weight;

            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                parent[v] = u;
                pq.emplace(new_dist, v);
            }
        }
    }

    std::cout << "Computed SOSP for objective " << objective_idx << std::endl;
}

// Serial Bellman-Ford algorithm for MOSP
void serialBellmanFord(Graph& graph, int source, int objective_idx, std::vector<int>& dist,
                       std::vector<int>& parent) {
    int n = graph.V;
    dist.assign(n, std::numeric_limits<int>::max());
    parent.assign(n, -1);

    if (source < 0 || source >= n) {
        std::cerr << "Invalid source vertex: " << source << std::endl;
        exit(1);
    }

    dist[source] = 0;
    parent[source] = source;

    bool changed;
    for (int iteration = 1; iteration < n; ++iteration) {
        changed = false;
        for (int u = 0; u < n; ++u) {
            for (const Edge& edge : graph.adj[u]) {
                int v = edge.dest;
                int weight = edge.weights.empty() ? 1 : edge.weights[objective_idx % edge.weights.size()];
                int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                               std::numeric_limits<int>::max() : dist[u] + weight;

                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    parent[v] = u;
                    changed = true;
                }
            }
        }
        std::cout << "Bellman-Ford iteration " << iteration
                  << ", changes: " << (changed ? "yes" : "no") << std::endl;
        if (!changed) {
            std::cout << "Bellman-Ford converged early at iteration " << iteration << std::endl;
            break;
        }
    }
}

// Update SOSP tree for edge insertions
void updateSOSP(Graph& graph, int objective_idx, std::vector<int>& dist,
                std::vector<int>& parent, const std::vector<EdgeModification>& insertions) {
    int n = graph.V;
    std::vector<std::vector<EdgeModification>> grouped_insertions(n);
    std::vector<int> marked(n, 0);
    std::vector<int> affected;

    // Step 0: Preprocessing
    for (const auto& ins : insertions) {
        if (ins.is_insertion) {
            grouped_insertions[ins.dest].push_back(ins);
        }
    }

    // Step 1: Process changed edges
    for (int v = 0; v < n; v++) {
        for (const auto& ins : grouped_insertions[v]) {
            int u = ins.src;
            Edge edge;
            edge.dest = v;
            edge.weights = ins.weights;
            graph.adj[u].push_back(edge);
            int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                           std::numeric_limits<int>::max() : dist[u] + ins.weights[objective_idx % ins.weights.size()];
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                parent[v] = u;
                affected.push_back(v);
                marked[v] = 1;
            }
        }
    }

    // Step 2: Propagate the update
    int iteration = 0;
    while (!affected.empty()) {
        iteration++;
        std::vector<int> next_affected;
        std::set<int> neighbors;

        for (int u : affected) {
            for (const auto& edge : graph.adj[u]) {
                neighbors.insert(edge.dest);
            }
        }

        for (int v : neighbors) {
            for (int u = 0; u < n; u++) {
                for (const auto& edge : graph.adj[u]) {
                    if (edge.dest == v && marked[u] == 1) {
                        int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                                       std::numeric_limits<int>::max() : dist[u] + edge.weights[objective_idx % edge.weights.size()];
                        if (new_dist < dist[v]) {
                            dist[v] = new_dist;
                            parent[v] = u;
                            if (marked[v] == 0) {
                                next_affected.push_back(v);
                                marked[v] = 1;
                            }
                        }
                    }
                }
            }
        }

        affected = next_affected;
        std::cout << "Insertion update iteration " << iteration << " for objective " << objective_idx
                  << ", affected vertices: " << affected.size()
                  << ", changes: " << (affected.empty() ? "no" : "yes") << std::endl;

        if (iteration > 100) {
            std::cout << "Insertion update reached iteration limit for objective " << objective_idx << std::endl;
            break;
        }
    }
}

// Create combined graph for MOSP
Graph createCombinedGraph(const std::vector<SOSPTree>& sosp_trees, int V, int k) {
    Graph combined_graph;
    combined_graph.V = V;
    combined_graph.adj.resize(V);

    std::map<std::pair<int, int>, int> edge_counts;
    for (const auto& tree : sosp_trees) {
        for (const auto& [u, v] : tree.edges) {
            edge_counts[{u, v}]++;
        }
    }

    for (const auto& [edge, count] : edge_counts) {
        int u = edge.first;
        int v = edge.second;
        Edge e;
        e.dest = v;
        e.weights = {k - count + 1};
        combined_graph.adj[u].push_back(e);
    }

    std::cout << "Created combined graph with " << edge_counts.size() << " edges" << std::endl;
    return combined_graph;
}

// Assign original weights to MOSP tree
void assignOriginalWeights(const Graph& original_graph, const std::vector<int>& mosp_parent,
                           std::vector<std::pair<int, int>>& mosp_edges,
                           std::vector<std::vector<int>>& mosp_weights) {
    mosp_edges.clear();
    mosp_weights.clear();
    int source = -1;
    for (int v = 0; v < original_graph.V; ++v) {
        if (mosp_parent[v] == v) {
            source = v;
        }
        if (mosp_parent[v] != -1 && v != mosp_parent[v]) {
            mosp_edges.emplace_back(mosp_parent[v], v);
        }
    }

    for (const auto& [u, v] : mosp_edges) {
        std::vector<int> weights;
        bool found = false;
        for (const auto& edge : original_graph.adj[u]) {
            if (edge.dest == v) {
                weights = edge.weights;
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Warning: Edge (" << u << "," << v << ") not found in original graph" << std::endl;
            weights.resize(original_graph.adj[u].empty() ? 1 : original_graph.adj[u][0].weights.size(), -1);
        }
        mosp_weights.push_back(weights);
    }
}

// Verify MOSP Pareto optimality
void verifyMOSP(const Graph& graph, const std::vector<int>& mosp_dist, const std::vector<int>& mosp_parent,
                int source, int num_objectives) {
    std::cout << "\nVerifying MOSP Pareto Optimality..." << std::endl;

    auto computePathObjectives = [&](const std::vector<int>& path) -> std::vector<int> {
        std::vector<int> objectives(num_objectives, 0);
        for (size_t i = 1; i < path.size(); ++i) {
            int u = path[i-1];
            int v = path[i];
            bool found = false;
            for (const auto& edge : graph.adj[u]) {
                if (edge.dest == v) {
                    for (int j = 0; j < num_objectives; ++j) {
                        objectives[j] += edge.weights[j % edge.weights.size()];
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Warning: Edge (" << u << "," << v << ") not found in path" << std::endl;
                return std::vector<int>(num_objectives, std::numeric_limits<int>::max());
            }
        }
        return objectives;
    };

    auto isDominated = [](const std::vector<int>& objectives1, const std::vector<int>& objectives2) -> bool {
        bool at_least_one_strictly_better = false;
        for (size_t i = 0; i < objectives1.size(); ++i) {
            if (objectives2[i] > objectives1[i]) {
                return false;
            }
            if (objectives2[i] < objectives1[i]) {
                at_least_one_strictly_better = true;
            }
        }
        return at_least_one_strictly_better;
    };

    std::function<void(int, int, std::vector<int>&, std::vector<std::vector<int>>&, int)> enumeratePaths;
    enumeratePaths = [&](int u, int dest, std::vector<int>& current_path, std::vector<std::vector<int>>& all_paths, int max_depth) {
        if (u == dest) {
            all_paths.push_back(current_path);
            return;
        }
        if (current_path.size() >= static_cast<size_t>(max_depth)) {
            return;
        }
        for (const auto& edge : graph.adj[u]) {
            int v = edge.dest;
            if (std::find(current_path.begin(), current_path.end(), v) == current_path.end()) {
                current_path.push_back(v);
                enumeratePaths(v, dest, current_path, all_paths, max_depth);
                current_path.pop_back();
            }
        }
    };

    int sample_count = std::min(20, graph.V);
    for (int dest = 0; dest < sample_count; ++dest) {
        if (mosp_dist[dest] == std::numeric_limits<int>::max()) {
            std::cout << "\nVertex " << dest << ": No path exists" << std::endl;
            continue;
        }

        std::vector<int> mosp_path;
        for (int v = dest; v != source && v != -1; v = mosp_parent[v]) {
            mosp_path.push_back(v);
            if (mosp_path.size() > static_cast<size_t>(graph.V)) {
                std::cout << "\nVertex " << dest << ": [loop detected in MOSP path]" << std::endl;
                break;
            }
        }
        mosp_path.push_back(source);
        std::reverse(mosp_path.begin(), mosp_path.end());

        std::vector<int> mosp_objectives = computePathObjectives(mosp_path);
        if (mosp_objectives[0] == std::numeric_limits<int>::max()) {
            std::cout << "\nVertex " << dest << ": Invalid MOSP path" << std::endl;
            continue;
        }

        std::cout << "\nVertex " << dest << " MOSP Path: ";
        for (size_t i = 0; i < mosp_path.size(); ++i) {
            std::cout << mosp_path[i];
            if (i < mosp_path.size() - 1) std::cout << " -> ";
        }
        std::cout << "\nObjectives (e.g., Travel Time, Fuel Consumption): ";
        for (int obj : mosp_objectives) {
            std::cout << obj << " ";
        }
        std::cout << std::endl;

        std::vector<std::vector<int>> alternative_paths;
        std::vector<int> current_path = {source};
        int max_depth = graph.V * 2;
        enumeratePaths(source, dest, current_path, alternative_paths, max_depth);

        bool is_pareto_optimal = true;
        for (const auto& alt_path : alternative_paths) {
            if (alt_path == mosp_path) continue;
            std::vector<int> alt_objectives = computePathObjectives(alt_path);
            if (alt_objectives[0] == std::numeric_limits<int>::max()) continue;
            if (isDominated(mosp_objectives, alt_objectives)) {
                is_pareto_optimal = false;
                std::cout << "Dominated by alternative path: ";
                for (size_t i = 0; i < alt_path.size(); ++i) {
                    std::cout << alt_path[i];
                    if (i < alt_path.size() - 1) std::cout << " -> ";
                }
                std::cout << "\nAlternative Objectives: ";
                for (int obj : alt_objectives) {
                    std::cout << obj << " ";
                }
                std::cout << std::endl;
            }
        }

        std::cout << "Status: " << (is_pareto_optimal ? "Pareto-optimal" : "Sub-optimal") << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Validate number of objectives
    int num_objectives = 2; // Default to bi-objective
    std::vector<int> sources = {0, 0};
    if (argc > 1) {
        try {
            num_objectives = std::stoi(argv[1]);
            if (num_objectives <= 0) {
                std::cerr << "Error: Number of objectives must be positive. Got: " << num_objectives << std::endl;
                return 1;
            }
            sources.clear();
            for (int i = 0; i < num_objectives; ++i) {
                sources.push_back(0);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid number of objectives provided: " << argv[1] << std::endl;
            return 1;
        }
    }

    std::cout << "Starting MOSP computation with " << num_objectives << " objectives" << std::endl;
    std::cout << "Source vertex: " << sources[0] << std::endl;

    // Create graph
    std::cout << "Creating hardcoded graph..." << std::endl;
    Graph graph = createHardcodedGraph(num_objectives);

    // Compute initial SOSP trees
    std::vector<SOSPTree> sosp_trees(num_objectives);
    double start_time = static_cast<double>(clock()) / CLOCKS_PER_SEC;
    for (int i = 0; i < num_objectives; ++i) {
        std::cout << "Computing initial SOSP for objective " << i << " using Dijkstra's..." << std::endl;
        serialSOSP(graph, sources[i], i, sosp_trees[i].dist, sosp_trees[i].parent);
        sosp_trees[i].edges.clear();
        for (int v = 0; v < graph.V; ++v) {
            if (sosp_trees[i].parent[v] != -1 && v != sources[i]) {
                sosp_trees[i].edges.emplace_back(sosp_trees[i].parent[v], v);
            }
        }
    }

    // Apply edge insertions
    std::vector<EdgeModification> insertions = {
        {1, 2, {3, 4}, true},
        {5, 7, {2, 3}, true}
    };
    std::cout << "Applying edge insertions..." << std::endl;

    for (int i = 0; i < num_objectives; ++i) {
        std::cout << "Updating SOSP for objective " << i << " after insertions..." << std::endl;
        updateSOSP(graph, i, sosp_trees[i].dist, sosp_trees[i].parent, insertions);
        sosp_trees[i].edges.clear();
        for (int v = 0; v < graph.V; ++v) {
            if (sosp_trees[i].parent[v] != -1 && v != sources[i]) {
                sosp_trees[i].edges.emplace_back(sosp_trees[i].parent[v], v);
            }
        }
    }

    // Create combined graph
    Graph combined_graph = createCombinedGraph(sosp_trees, graph.V, num_objectives);

    // Compute MOSP
    std::vector<int> mosp_dist, mosp_parent;
    std::cout << "Computing MOSP on combined graph using Bellman-Ford..." << std::endl;
    serialBellmanFord(combined_graph, sources[0], 0, mosp_dist, mosp_parent);
    double end_time = static_cast<double>(clock()) / CLOCKS_PER_SEC;

    // Assign original weights
    std::vector<std::pair<int, int>> mosp_edges;
    std::vector<std::vector<int>> mosp_weights;
    assignOriginalWeights(graph, mosp_parent, mosp_edges, mosp_weights);

    // Verify MOSP
    verifyMOSP(graph, mosp_dist, mosp_parent, sources[0], num_objectives);

    // Output results
    std::cout << "\nMOSP Results (Bellman-Ford):" << std::endl;
    std::cout << "Total execution time: " << (end_time - start_time) << " seconds" << std::endl;
    std::cout << "\nMOSP Tree Edges:" << std::endl;
    for (size_t i = 0; i < mosp_edges.size(); ++i) {
        auto [u, v] = mosp_edges[i];
        std::cout << "Edge (" << u << "," << v << ") weights: ";
        for (int w : mosp_weights[i]) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }
    int reachable = 0;
    for (int d : mosp_dist) {
        if (d != std::numeric_limits<int>::max()) {
            reachable++;
        }
    }
    std::cout << "\nReachable vertices in MOSP: " << reachable << "/" << graph.V << std::endl;

    int sample_count = std::min(20, graph.V);
    std::cout << "\nShortest paths (first " << sample_count << " vertices):" << std::endl;
    std::cout << "Vertex\tDistance\tPath" << std::endl;
    for (int i = 0; i < sample_count; ++i) {
        if (mosp_dist[i] != std::numeric_limits<int>::max()) {
            std::cout << i << "\t" << mosp_dist[i] << "\t\t";
            std::vector<int> path;
            for (int v = i; v != sources[0] && v != -1; v = mosp_parent[v]) {
                path.push_back(v);
                if (path.size() > graph.V) {
                    std::cout << "[loop detected]";
                    break;
                }
            }
            path.push_back(sources[0]);
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                std::cout << *it << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << i << "\tINF\t\tNo path" << std::endl;
        }
    }

    return 0;
}