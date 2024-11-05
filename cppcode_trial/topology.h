#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <utility> // for std::pair
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/cycle.hpp> // if available, else implement custom cycle detection
#include <boost/graph/undirected_dfs.hpp>

#include "tissue_types.h" // Include shared type definitions

using namespace std;
using namespace Eigen;

// Adjust bounds of vertices within range -L to L
Vertices verticesInBounds(const Vertices& vertices, double L);

// Removes invalid edges (-1 entries) from ridges
Ridges removeMinus(const Ridges& ridges);

// Create adjacency matrix for region R using vertex ridges
MatrixXi adjMat(const Region& R, const Ridges& ridges);

// Flatten nested vector into a single vector
template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec);

// Reorder elements based on graph cycles
std::vector<int> rearrange(int n, const AdjacencyList& binMat, bool toPrint = false);

// Reorder regions based on adjacency matrix
Regions rearrangeRegions(const Regions& regions, const PointRegion& pointRegion, const Ridges& ridges);

// Find the number of sides for a cell in a Voronoi tessellation
int nsidesVor(const PointRegion& pointRegion, const Regions& regions, int i);

// Functions to find neighborhood elements
Region findCenterRegion(const Regions& regions, const PointRegion& pointRegion, int center);
vector<int> findVertexNeighbourVertices(const Ridges& ridges, int vertex);
pair<vector<int>, vector<int>> findVertexNeighbourCenters(const Regions& regions, const PointRegion& pointRegion, int vertex);
pair<vector<int>, vector<int>> findVertexNeighbour(const Regions& regions, const PointRegion& pointRegion, const Ridges& ridges, int vertex);
vector<int> findCenterNeighbourCenter(const Regions& regions, const PointRegion& pointRegion, int center);
vector<int> findBoundaryVertices(int nVertices, const Ridges& ridges);
vector<int> findBoundaryVerticesSquare(int nVertices, const Ridges& ridges);
vector<int> findWoundBoundary(const Regions& regions, const PointRegion& pointRegion, int woundLoc);

#endif // TOPOLOGY_H