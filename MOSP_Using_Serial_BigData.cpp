#include <iostream>
#include <vector>
#include <limits>
#include <map>
#include <set>
//#include <mpi.h>
//#include <omp.h>
#include <metis.h>
#include <string>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

#define ROOT 0

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

// Read graph from weighted_graph_usa.txt
Graph readGraphFromFile(int num_objectives) {
    Graph graph;
    std::vector<std::tuple<int, int, std::vector<int>>> edges;
    std::map<long long, int> vertex_map; // Map original IDs to indices
    int vertex_index = 0;
    int min_weight = std::numeric_limits<int>::max();
    int max_weight = std::numeric_limits<int>::min();

    std::ifstream file("weighted_graph_usa.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weighted_graph_usa.txt" << std::endl;
        std::exit(1);
    }

    std::string line; int maxit =0;
    while (std::getline(file, line)) {
    
    	//maxit++;
    	//if(maxit>100000) break;
        std::istringstream iss(line);
        long long src, dest;
        int weight1, weight2 = 0; // Default second weight to 0 if not provided
        if (!(iss >> src >> dest >> weight1)) {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
            continue;
        }
        // Read second weight if available (for num_objectives >= 2)
        if (num_objectives >= 2) {
            if (!(iss >> weight2)) {
                weight2 = weight1; // Use first weight if second is missing
            }
        }

        // Update min/max weights
        min_weight = std::min({min_weight, weight1, weight2});
        max_weight = std::max({max_weight, weight1, weight2});

        // Map vertices to indices
        if (vertex_map.find(src) == vertex_map.end()) {
            vertex_map[src] = vertex_index++;
        }
        if (vertex_map.find(dest) == vertex_map.end()) {
            vertex_map[dest] = vertex_index++;
        }

        // Add edge
        std::vector<int> weights = {weight1, weight2};
        edges.emplace_back(vertex_map[src], vertex_map[dest], weights);
    }
    file.close();

    graph.V = vertex_map.size();
    graph.adj.resize(graph.V);

    // Ensure num_objectives is at least 1
    int effective_objectives = std::max(1, num_objectives);

    // Populate adjacency list
    for (auto& [u, v, w] : edges) {
        Edge edge;
        edge.dest = v;
        edge.weights = w;
        if (edge.weights.size() < static_cast<size_t>(effective_objectives)) {
            edge.weights.resize(effective_objectives, w.empty() ? 1 : w[0]);
        } else if (edge.weights.size() > static_cast<size_t>(effective_objectives)) {
            edge.weights.resize(effective_objectives);
        }
        graph.adj[u].push_back(edge);
    }

    std::cout << "Read graph from weighted_graph_usa.txt with " << graph.V << " vertices, "
              << edges.size() << " edges, and " << effective_objectives << " objectives" << std::endl;
    std::cout << "Weight range: [" << min_weight << ", " << max_weight << "]" << std::endl;

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
        std::exit(1);
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
                adjwgt[edge_count] = edge.weights.empty() ? 1 : edge.weights[0];
                edge_count++;
            }
        }
        xadj[i + 1] = edge_count;
    }

    part.resize(nvtxs);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    std::cout << "Starting METIS partitioning with " << nparts << " parts..." << std::endl;

    idx_t metis_nparts = nparts;
    idx_t objval;

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, nullptr, adjwgt,
                                  &metis_nparts, nullptr, nullptr, options, &objval, part.data());

    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed with error code: " << ret << std::endl;
        std::exit(1);
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

// Serial Dijkstra's algorithm for SOSP
void parallelSOSP(Graph& graph, int source, int objective_idx, std::vector<int>& dist,
                  std::vector<int>& parent) {
    int n = graph.V;
    dist.assign(n, std::numeric_limits<int>::max());
    parent.assign(n, -1);

    if (source < 0 || source >= n) {
        std::cerr << "Invalid source vertex: " << source << std::endl;
        std::exit(1);
    }

    dist[source] = 0;
    parent[source] = source;

    std::vector<bool> visited(n, false);
    std::vector<int> active_vertices = {source};
    int iteration = 0;

    while (!active_vertices.empty()) {
        iteration++;
        std::vector<int> next_active;

        // Serial loop over active vertices
        for (size_t i = 0; i < active_vertices.size(); i++) {
            int u = active_vertices[i];
            if (visited[u]) continue;
            visited[u] = true;

            for (const Edge& edge : graph.adj[u]) {
                int v = edge.dest;
                int weight = edge.weights.empty() ? 1 : edge.weights[objective_idx % edge.weights.size()];
                int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                               std::numeric_limits<int>::max() : dist[u] + weight;

                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    parent[v] = u;
                    if (!visited[v]) {
                        next_active.push_back(v);
                    }
                }
            }
        }

        std::set<int> unique_active(next_active.begin(), next_active.end());
        active_vertices.assign(unique_active.begin(), unique_active.end());

        std::cout << "Dijkstra iteration " << iteration << " for objective " << objective_idx
                  << ", active vertices: " << active_vertices.size() << std::endl;

        if (iteration > n) {
            std::cout << "Reached iteration limit for objective " << objective_idx << std::endl;
            break;
        }
    }
}

// Serial Bellman-Ford algorithm for MOSP
void parallelBellmanFord(Graph& graph, int source, int objective_idx, std::vector<int>& dist,
                         std::vector<int>& parent) {
    int n = graph.V;
    dist.assign(n, std::numeric_limits<int>::max());
    parent.assign(n, -1);

    if (source < 0 || source >= n) {
        std::cerr << "Invalid source vertex: " << source << std::endl;
        std::exit(1);
    }

    dist[source] = 0;
    parent[source] = source;

    bool changed = false;
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

// Update SOSP tree for edge insertions and deletions
void updateSOSP(Graph& graph, int objective_idx, std::vector<int>& dist,
                std::vector<int>& parent, const std::vector<EdgeModification>& modifications) {
    int n = graph.V;
    std::vector<int> marked(n, 0);
    std::vector<int> affected;

    // Step 1: Process deletions
    {
        std::vector<int> thread_affected;
        for (size_t i = 0; i < modifications.size(); i++) {
            const auto& mod = modifications[i];
            if (mod.is_insertion) continue;
            int u = mod.src;
            int v = mod.dest;

            {
                auto& adj = graph.adj[u];
                auto it = std::find_if(adj.begin(), adj.end(),
                    [v](const Edge& e) { return e.dest == v; });
                if (it != adj.end()) {
                    adj.erase(it);
                } else {
                    std::cerr << "Warning: Edge (" << u << "," << v << ") not found for deletion" << std::endl;
                }
            }

            if (parent[v] == u && v < n) {
                dist[v] = std::numeric_limits<int>::max();
                parent[v] = -1;
                thread_affected.push_back(v);
                marked[v] = 1;
            }
        }
        {
            affected.insert(affected.end(), thread_affected.begin(), thread_affected.end());
        }
    }

    bool global_changed = true;
    int deletion_iteration = 0;
    while (global_changed && !affected.empty()) {
        global_changed = false;
        deletion_iteration++;
        std::vector<int> next_affected;
        std::vector<int> neighbors;

        {
            std::vector<int> thread_neighbors;
            for (size_t i = 0; i < affected.size(); i++) {
                int u = affected[i];
                if (u >= n) continue;
                for (const auto& edge : graph.adj[u]) {
                    thread_neighbors.push_back(edge.dest);
                }
            }
            {
                neighbors.insert(neighbors.end(), thread_neighbors.begin(), thread_neighbors.end());
            }
        }

        std::set<int> unique_neighbors(neighbors.begin(), neighbors.end());
        neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());

        {
            std::vector<int> thread_affected;
            std::vector<std::tuple<int, int, int>> thread_updates;
            for (size_t i = 0; i < neighbors.size(); i++) {
                int v = neighbors[i];
                if (v >= n) continue;

                bool reset = false;
                if (parent[v] != -1 && parent[v] < n && marked[parent[v]] == 1) {
                    dist[v] = std::numeric_limits<int>::max();
                    parent[v] = -1;
                    reset = true;
                    thread_affected.push_back(v);
                    marked[v] = 1;
                }

                for (int u = 0; u < n; u++) {
                    for (const auto& edge : graph.adj[u]) {
                        if (edge.dest == v) {
                            int weight = edge.weights.empty() ? 1 : edge.weights[objective_idx % edge.weights.size()];
                            int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                                           std::numeric_limits<int>::max() : dist[u] + weight;
                            if (new_dist < dist[v]) {
                                thread_updates.emplace_back(v, new_dist, u);
                                if (!reset && marked[v] == 0) {
                                    thread_affected.push_back(v);
                                    marked[v] = 1;
                                }
                            }
                        }
                    }
                }
            }
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

        std::cout << "Deletion update iteration " << deletion_iteration << " for objective " << objective_idx
                  << ", affected vertices: " << affected.size()
                  << ", changes: " << (global_changed ? "yes" : "no") << std::endl;

        if (deletion_iteration > 100) {
            std::cout << "Deletion update reached iteration limit for objective " << objective_idx << std::endl;
            break;
        }
    }

    std::vector<std::vector<EdgeModification>> grouped_insertions(n);
    for (const auto& mod : modifications) {
        if (mod.is_insertion) {
            grouped_insertions[mod.dest].push_back(mod);
        }
    }

    {
        std::vector<int> thread_affected;
        for (int v = 0; v < n; v++) {
            for (const auto& ins : grouped_insertions[v]) {
                int u = ins.src;
                Edge edge;
                edge.dest = v;
                edge.weights = ins.weights;
                {
                    graph.adj[u].push_back(edge);
                }
                int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                               std::numeric_limits<int>::max() : dist[u] + ins.weights[objective_idx % ins.weights.size()];
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    parent[v] = u;
                    thread_affected.push_back(v);
                    marked[v] = 1;
                }
            }
        }
        {
            affected.insert(affected.end(), thread_affected.begin(), thread_affected.end());
        }
    }

    global_changed = true;
    int insertion_iteration = 0;
    while (global_changed) {
        global_changed = false;
        insertion_iteration++;
        std::vector<int> next_affected;
        std::vector<int> neighbors;

        {
            std::vector<int> thread_neighbors;
            for (size_t i = 0; i < affected.size(); i++) {
                int u = affected[i];
                if (u >= n) continue;
                for (const auto& edge : graph.adj[u]) {
                    thread_neighbors.push_back(edge.dest);
                }
            }
            {
                neighbors.insert(neighbors.end(), thread_neighbors.begin(), thread_neighbors.end());
            }
        }

        std::set<int> unique_neighbors(neighbors.begin(), neighbors.end());
        neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());

        {
            std::vector<int> thread_affected;
            std::vector<std::tuple<int, int, int>> thread_updates;
            for (size_t i = 0; i < neighbors.size(); i++) {
                int v = neighbors[i];
                if (v >= n) continue;
                for (int u = 0; u < n; u++) {
                    for (const auto& edge : graph.adj[u]) {
                        if (edge.dest == v && marked[u] == 1) {
                            int new_dist = (dist[u] == std::numeric_limits<int>::max()) ?
                                           std::numeric_limits<int>::max() : dist[u] + edge.weights[objective_idx % edge.weights.size()];
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

        std::cout << "Insertion update iteration " << insertion_iteration << " for objective " << objective_idx
                  << ", affected vertices: " << affected.size()
                  << ", changes: " << (global_changed ? "yes" : "no") << std::endl;

        if (insertion_iteration > 100) {
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
    // Timing variables
    double graph_load_time = 0.0;
    double metis_time = 0.0;
    double sosp_time = 0.0;
    double mosp_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_time, end_time;

    //MPI_Init(&argc, &argv);
    //int rank, size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    Graph graph;
    std::vector<std::tuple<int, int, std::vector<int>>> raw_edges; // Store edges for modifications
    int min_weight = 1, max_weight = 10; // Defaults if file read fails

    // Start graph loading time
    start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Reading graph from weighted_graph_usa.txt..." << std::endl;
    graph = readGraphFromFile(num_objectives);

    // Collect raw edges for modification selection
    for (int u = 0; u < graph.V; ++u) {
        for (const auto& edge : graph.adj[u]) {
            raw_edges.emplace_back(u, edge.dest, edge.weights);
            min_weight = std::min(min_weight, edge.weights[0]);
            max_weight = std::max(max_weight, edge.weights[0]);
            if (num_objectives >= 2) {
                min_weight = std::min(min_weight, edge.weights[1]);
                max_weight = std::max(max_weight, edge.weights[1]);
            }
        }
    }

    // End graph loading time
    end_time = std::chrono::high_resolution_clock::now();
    graph_load_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

    // Start METIS partitioning time
    start_time = std::chrono::high_resolution_clock::now();

    // Partition graph (optional, set to 1 part for serial)
    std::vector<int> part(graph.V, 0); // All vertices in partition 0
    // Uncomment to use METIS with 1 part:
    // std::cout << "Partitioning graph..." << std::endl;
    // partitionGraph(graph, 1, part);

    // End METIS partitioning time
    end_time = std::chrono::high_resolution_clock::now();
    metis_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

    // Start SOSP computation time
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<SOSPTree> sosp_trees(num_objectives);
    for (int i = 0; i < num_objectives; ++i) {
        std::cout << "Computing initial SOSP for objective " << i << " using Dijkstra's..." << std::endl;
        parallelSOSP(graph, sources[i], i, sosp_trees[i].dist, sosp_trees[i].parent);

        sosp_trees[i].edges.clear();
        for (int v = 0; v < graph.V; ++v) {
            if (sosp_trees[i].parent[v] != -1 && v != sources[i]) {
                sosp_trees[i].edges.emplace_back(sosp_trees[i].parent[v], v);
            }
        }
    }

    std::vector<EdgeModification> modifications;
    std::cout << "Generating edge modifications (insertions and deletions)..." << std::endl;

    // Select 2 edges for deletion
    if (raw_edges.size() < 2) {
        std::cerr << "Error: Not enough edges to select for deletion" << std::endl;
        std::exit(1);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> edge_dist(0, raw_edges.size() - 1);
    std::set<int> selected_indices;

    // Pick 2 unique edges for deletion
    while (selected_indices.size() < 2) {
        selected_indices.insert(edge_dist(gen));
    }

    for (int idx : selected_indices) {
        auto [u, v, w] = raw_edges[idx];
        modifications.push_back({u, v, {}, false});
        std::cout << "Selected for deletion: (" << u << "," << v << ")" << std::endl;
    }

    // Generate 2 new edges for insertion
    std::uniform_int_distribution<> vertex_dist(0, graph.V - 1);
    std::uniform_int_distribution<> weight_dist(min_weight, max_weight);
    for (int i = 0; i < 2; ++i) {
        int u, v;
        bool exists;
        do {
            u = vertex_dist(gen);
            v = vertex_dist(gen);
            if (u == v) continue;
            exists = false;
            for (const auto& edge : graph.adj[u]) {
                if (edge.dest == v) {
                    exists = true;
                    break;
                }
            }
        } while (exists); // Ensure edge doesn't already exist

        std::vector<int> weights(num_objectives);
        for (int j = 0; j < num_objectives; ++j) {
            weights[j] = weight_dist(gen);
        }
        modifications.push_back({u, v, weights, true});
        std::cout << "Selected for insertion: (" << u << "," << v << ") with weights: ";
        for (int w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Applying edge modifications (insertions and deletions)..." << std::endl;

    for (int i = 0; i < num_objectives; ++i) {
        std::cout << "Updating SOSP for objective " << i << " after modifications..." << std::endl;
        updateSOSP(graph, i, sosp_trees[i].dist, sosp_trees[i].parent, modifications);

        sosp_trees[i].edges.clear();
        for (int v = 0; v < graph.V; ++v) {
            if (sosp_trees[i].parent[v] != -1 && v != sources[i]) {
                sosp_trees[i].edges.emplace_back(sosp_trees[i].parent[v], v);
            }
        }
    }

    // End SOSP computation time
    end_time = std::chrono::high_resolution_clock::now();
    sosp_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

    // Start MOSP computation time
    start_time = std::chrono::high_resolution_clock::now();

    Graph combined_graph;
    combined_graph = createCombinedGraph(sosp_trees, graph.V, num_objectives);

    std::vector<int> mosp_dist, mosp_parent;
    std::cout << "Computing MOSP on combined graph using Bellman-Ford..." << std::endl;
    parallelBellmanFord(combined_graph, sources[0], 0, mosp_dist, mosp_parent);

    std::vector<std::pair<int, int>> mosp_edges;
    std::vector<std::vector<int>> mosp_weights;
    assignOriginalWeights(graph, mosp_parent, mosp_edges, mosp_weights);

    //verifyMOSP(graph, mosp_dist, mosp_parent, sources[0], num_objectives);

    // End MOSP computation time
    end_time = std::chrono::high_resolution_clock::now();
    mosp_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

    std::cout << "\nMOSP Results (Bellman-Ford):" << std::endl;
    // Print individual timing results
    std::cout << "\n=== Execution Times ===" << std::endl;
    std::cout << "Graph Loading Time: " << graph_load_time << " seconds" << std::endl;
    std::cout << "METIS Partitioning Time: " << metis_time << " seconds" << std::endl;
    std::cout << "SOSP Computation Time : " << sosp_time << " seconds" << std::endl;
    std::cout << "MOSP Computation Time : " << mosp_time << " seconds" << std::endl;
    std::cout << "Total Execution Time: " << (graph_load_time + metis_time + sosp_time + mosp_time) << " seconds" << std::endl;

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

    //MPI_Finalize();
    return 0;
}
