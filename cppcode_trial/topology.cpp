#include "topology.h"
#include "tissue_types.h" // include type definitions
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <set>
#include <stack>

using namespace std;
using namespace Eigen;
using namespace boost;

Vertices verticesInBounds(const Vertices& vertices, double L) {
    Vertices boundedVertices = vertices;
    for (auto& vertex : boundedVertices) {
        for (double& coord : vertex) {
            coord = max(-L, min(coord, L));
        }
    }
    return boundedVertices;
}

Ridges removeMinus(const Ridges& ridges) {
    Ridges cleanedRidges;
    for (const auto& ridge : ridges) {
        if (find(ridge.begin(), ridge.end(), -1) == ridge.end()) {
            cleanedRidges.push_back(ridge);
        }
    }
    return cleanedRidges;
}

MatrixXi adjMat(const Region& R, const Ridges& ridges) {
    int n = R.size();
    MatrixXi binMat = MatrixXi::Zero(n, n);
    unordered_map<int, int> vertexIndex;
    for (int i = 0; i < n; ++i) vertexIndex[R[i]] = i;

    for (int vi : R) {
        vector<int> neighbors = findVertexNeighbourVertices(ridges, vi);
        int loc_i = vertexIndex[vi];
        for (int vj : neighbors) {
            if (vertexIndex.count(vj)) {
                int loc_j = vertexIndex[vj];
                binMat(loc_i, loc_j) = 1;
            }
        }
    }
    return binMat;
}

template<typename T>
vector<T> flatten(const vector<vector<T>>& vec) {
    vector<T> flatVec;
    for (const auto& subVec : vec) {
        flatVec.insert(flatVec.end(), subVec.begin(), subVec.end());
    }
    return flatVec;
}

// Type definitions for the Boost graph
typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

// Function to detect cycles
void find_cycles(const Graph& g, std::vector<std::vector<int>>& cycles) {
    std::vector<int> parent(num_vertices(g), -1); // parent array for DFS
    std::vector<bool> visited(num_vertices(g), false);

    for (auto u : make_iterator_range(vertices(g))) {
        if (!visited[u]) {
            // Stack to store the DFS traversal
            std::stack<std::pair<Vertex, Vertex>> stack;
            stack.push({u, u});

            while (!stack.empty()) {
                Vertex v, p;
                std::tie(v, p) = stack.top();
                stack.pop();

                if (visited[v]) {
                    // Cycle detected: trace back through parents to form the cycle
                    std::vector<int> cycle;
                    Vertex curr = v;
                    while (curr != p && curr != -1) {
                        cycle.push_back(curr);
                        curr = parent[curr];
                    }
                    cycle.push_back(p); // complete the cycle
                    cycles.push_back(cycle);
                } else {
                    visited[v] = true;
                    parent[v] = p;

                    for (auto adj : make_iterator_range(adjacent_vertices(v, g))) {
                        if (adj != p) { // Avoid traversing back to parent
                            stack.push({adj, v});
                        }
                    }
                }
            }
        }
    }
}

// Rearrange function using Boost
std::vector<int> rearrange(int n, const AdjacencyList& binMat, bool toPrint) {
    Graph g(n);

    // Populate the graph based on adjacency matrix
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (binMat[i][j] == 1) {
                add_edge(i, j, g);
            }
        }
    }

    // Find cycles in the graph
    std::vector<std::vector<int>> cycles;
    find_cycles(g, cycles);

    if (cycles.empty()) {
        // If no cycle is found, return a default ordering
        std::vector<int> default_order(n);
        std::iota(default_order.begin(), default_order.end(), 0);
        return default_order;
    }

    // Sort cycles to find the smallest, akin to Python's sorted(cycles)[0]
    std::sort(cycles.begin(), cycles.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return a.size() < b.size();
    });

    std::vector<int> rearranged_cycle = cycles.front(); // smallest cycle

    if (toPrint) {
        std::cout << "Rearranged cycle: ";
        for (int v : rearranged_cycle) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    return rearranged_cycle;
}

Regions rearrangeRegions(const Regions& regions, const PointRegion& pointRegion, const Ridges& ridges) {
    Regions newRegions;
    for (int c : pointRegion) {
        Region R = findCenterRegion(regions, pointRegion, c);
        vector<int> rearrangeLoc = rearrange(R.size(), adjMat(R, ridges));
        Region reorderedR;
        for (int idx : rearrangeLoc) reorderedR.push_back(R[idx]);
        newRegions.push_back(reorderedR);
    }
    return newRegions;
}

int nsidesVor(const PointRegion& pointRegion, const Regions& regions, int i) {
    return findCenterRegion(regions, pointRegion, i).size();
}

Region findCenterRegion(const Regions& regions, const PointRegion& pointRegion, int center) {
    Region R = regions[pointRegion[center]];
    R.erase(remove(R.begin(), R.end(), -1), R.end()); // remove -1 elements
    return R;
}

vector<int> findVertexNeighbourVertices(const Ridges& ridges, int vertex) {
    set<int> neighbors;
    for (const auto& ridge : ridges) {
        if (find(ridge.begin(), ridge.end(), vertex) != ridge.end()) {
            for (int v : ridge) {
                if (v != vertex && v != -1) neighbors.insert(v);
            }
        }
    }
    return vector<int>(neighbors.begin(), neighbors.end());
}

pair<vector<int>, vector<int>> findVertexNeighbourCenters(const Regions& regions, const PointRegion& pointRegion, int vertex) {
    vector<int> listRegions, listCenters;
    for (size_t i = 0; i < regions.size(); ++i) {
        if (find(regions[i].begin(), regions[i].end(), vertex) != regions[i].end()) {
            listRegions.push_back(i);
            auto locPoints = find(pointRegion.begin(), pointRegion.end(), i);
            if (locPoints != pointRegion.end()) listCenters.push_back(distance(pointRegion.begin(), locPoints));
        }
    }
    return {listRegions, listCenters};
}

pair<vector<int>, vector<int>> findVertexNeighbour(const Regions& regions, const PointRegion& pointRegion, const Ridges& ridges, int vertex) {
    auto centers = findVertexNeighbourCenters(regions, pointRegion, vertex);
    auto vertices = findVertexNeighbourVertices(ridges, vertex);
    return {centers.second, vertices};
}

vector<int> findCenterNeighbourCenter(const Regions& regions, const PointRegion& pointRegion, int center) {
    vector<int> neighbors;
    Region R = findCenterRegion(regions, pointRegion, center);
    for (int v : R) {
        auto [_, L_c] = findVertexNeighbourCenters(regions, pointRegion, v);
        neighbors.insert(neighbors.end(), L_c.begin(), L_c.end());
    }
    neighbors.erase(remove(neighbors.begin(), neighbors.end(), center), neighbors.end());
    sort(neighbors.begin(), neighbors.end());
    neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    return neighbors;
}

vector<int> findBoundaryVertices(int nVertices, const Ridges& ridges) {
    vector<int> boundary;
    vector<int> degree(nVertices, 0);
    for (const auto& ridge : ridges) {
        if (ridge[0] != -1 && ridge[1] != -1) {
            degree[ridge[0]]++;
            degree[ridge[1]]++;
        }
    }
    for (int i = 0; i < nVertices; ++i) {
        if (degree[i] < 3) boundary.push_back(i);
    }
    return boundary;
}

vector<int> findBoundaryVerticesSquare(int nVertices, const Ridges& ridges) {
    vector<int> boundary;
    vector<int> degree(nVertices, 0);
    for (const auto& ridge : ridges) {
        if (ridge[0] != -1 && ridge[1] != -1) {
            degree[ridge[0]]++;
            degree[ridge[1]]++;
        }
    }
    for (int i = 0; i < nVertices; ++i) {
        if (degree[i] < 4) boundary.push_back(i);
    }
    return boundary;
}

vector<int> findWoundBoundary(const Regions& regions, const PointRegion& pointRegion, int woundLoc) {
    return findCenterRegion(regions, pointRegion, woundLoc);
}
