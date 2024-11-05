// tissue_types.h
#ifndef TISSUE_TYPES_H
#define TISSUE_TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <unordered_set>

// Alias for representing 2D points, commonly used for vertices
using Point2D = Eigen::Vector2d;

// Alias for holding a collection of 2D vertices (rows are points)
using Vertices = std::vector<Point2D>;

// Alias for representing a region, where each region is a collection of vertex indices
using Region = std::vector<int>;

// Alias for a collection of regions
using Regions = std::vector<Region>;

// Alias for representing point-region mapping (where each point has a corresponding region index)
using PointRegion = std::vector<int>;

// Alias for a collection of ridge connections (pairs of vertex indices)
using Ridges = std::vector<std::pair<int, int>>;

// Alias for representing vertex neighbors (adjacency lists of vertex indices)
using AdjacencyList = std::vector<std::unordered_set<int>>;

// Type alias for holding multiple centers (e.g., cell centers) as a collection of 2D points
using Centers = std::vector<Point2D>;

// Additional vector types for results (such as perimeters, areas, and strain)
using Perimeters = std::vector<double>;
using Areas = std::vector<double>;
using StrainData = Eigen::VectorXd; // For strain calculations

#endif // TISSUE_TYPES_H
