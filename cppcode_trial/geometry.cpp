#include "geometry.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric> // for std::accumulate
#include "topology.h"

/**
 * @brief Calculates the perimeter of a Voronoi cell.
 * 
 * @param point_region List of point-to-region indices.
 * @param regions List of all regions.
 * @param vertices List of vertices in the Voronoi diagram.
 * @param ridges List of ridge connections (edges between regions).
 * @param i Index of the target cell.
 * @return Calculated perimeter of the cell.
 */
double perimeter_vor(const PointRegion& point_region, const Regions& regions, 
                     const Vertices& vertices, const Ridges& ridges, int i) {
    std::vector<int> R = find_center_region(regions, point_region, i);
    
    double P = 0;
    if (R.size() > 2) {
        auto adjMatrix = adj_mat(R, ridges);
        std::vector<int> rearrange_loc = rearrange(R.size(), adjMatrix);
        Vertices V;

        if (rearrange_loc.size() == R.size()) {
            V = vertices; // Adjusted to take from vertices
            // Rearranging vertices according to rearrange_loc
            for (int idx : rearrange_loc) {
                V.push_back(vertices[idx]);
            }
        } else {
            V = vertices; // Using original vertices
        }

        for (int j = 0; j < V.size(); ++j) {
            P += (V[(j + 1) % V.size()] - V[j]).norm(); // Assuming V is a vector of 2D points
        }
    }
    return P;
}

/**
 * @brief Calculates the area of a Voronoi cell.
 */
double area_vor(const PointRegion& point_region, const Regions& regions, 
                const Vertices& vertices, const Ridges& ridges, int i, bool to_print) {
    std::vector<int> R = find_center_region(regions, point_region, i);
    double A1 = 0;

    if (R.size() > 2) {
        auto adjMatrix = adj_mat(R, ridges);
        std::vector<int> rearrange_loc = rearrange(R.size(), adjMatrix, to_print);
        Vertices V;

        if (rearrange_loc.size() == R.size()) {
            V = vertices; // Adjusted to take from vertices
            // Rearranging vertices according to rearrange_loc
            for (int idx : rearrange_loc) {
                V.push_back(vertices[idx]);
            }
        } else {
            V = vertices; // Using original vertices
        }

        for (int j = 0; j < V.size(); ++j) {
            A1 += V[j][0] * V[(j + 1) % V.size()][1] - V[j][1] * V[(j + 1) % V.size()][0];
        }
    }
    return 0.5 * std::abs(A1);
}

/**
 * @brief Calculates the perimeters of multiple Voronoi cells.
 */
std::vector<double> perimeters_vor(const PointRegion& point_region, const Regions& regions, 
                                   const Vertices& vertices, const Ridges& ridges, const std::vector<int>& list_i) {
    std::vector<double> perimeters;
    for (int i : list_i) {
        perimeters.push_back(perimeter_vor(point_region, regions, vertices, ridges, i));
    }
    return perimeters;
}

/**
 * @brief Calculates the areas of multiple Voronoi cells.
 */
std::vector<double> areas_vor(const PointRegion& point_region, const Regions& regions, 
                              const Vertices& vertices, const Ridges& ridges, const std::vector<int>& list_i, bool to_print) {
    std::vector<double> areas;
    for (int i : list_i) {
        areas.push_back(area_vor(point_region, regions, vertices, ridges, i, to_print));
    }
    return areas;
}

/**
 * @brief Calculates area evolution over time for a specific cell index.
 */
std::vector<double> area_time(const std::vector<PointRegion>& PointRegion, const std::vector<Regions>& Regions, 
                              const std::vector<Vertices>& Vertices, const std::vector<Ridges>& Ridges, int wloc, int NIter) {
    std::vector<double> area_list;
    for (int i = 0; i < NIter; ++i) {
        double area = area_vor(PointRegion[i], Regions[i], Vertices[i], Ridges[i], wloc);
        area_list.push_back(area);
    }
    return area_list;
}

/**
 * @brief Calculates perimeter evolution over time for a specific cell index.
 */
std::vector<double> perimeter_time(const std::vector<PointRegion>& PointRegion, const std::vector<Regions>& Regions, 
                                   const std::vector<Vertices>& Vertices, const std::vector<Ridges>& Ridges, int wloc, int NIter) {
    std::vector<double> perimeter_list;
    for (int i = 0; i < NIter; ++i) {
        double perimeter = perimeter_vor(PointRegion[i], Regions[i], Vertices[i], Ridges[i], wloc);
        perimeter_list.push_back(perimeter);
    }
    return perimeter_list;
}

/**
 * @brief Computes the average number of neighbors for a given cell and its neighbors.
 */
std::pair<std::vector<int>, std::vector<double>> shape_neighbour(const Regions& regions, const PointRegion& pregions) {
    std::vector<int> lreg;
    std::vector<double> lneigh;
    
    for (int c : pregions) {
        lreg.push_back(find_center_region(regions, pregions, c).size());
        auto Nc = find_center_neighbour_center(regions, pregions, c);
        
        std::vector<int> neigh_sizes;
        for (int k : Nc) {
            neigh_sizes.push_back(find_center_region(regions, pregions, k).size());
        }
        
        double average_neigh_size = std::accumulate(neigh_sizes.begin(), neigh_sizes.end(), 0.0) / neigh_sizes.size();
        lneigh.push_back(average_neigh_size);
    }
    return {lreg, lneigh};
}

/**
 * @brief Calculates the shape tensor for each cell in the tissue network.
 */
std::vector<std::vector<double>> shape_tensor(const Regions& regions, const PointRegion& point_region, 
                                              const Vertices& vertices, const Vertices& centers) {
    std::vector<std::vector<double>> S;
    
    for (int alpha : point_region) {
        int Nv = regions[alpha].size();
        auto loc_cell = centers[alpha]; // Assuming centers are also stored in Vertices format
        
        std::vector<double> Sa(4, 0.0); // [Sa[0][0], Sa[1][1], Sa[0][1], Sa[1][0]]
        
        for (int i = 0; i < Nv; ++i) {
            auto ria = vertices[regions[alpha][i]] - loc_cell;
            Sa[0] += ria[0] * ria[0];
            Sa[3] += ria[1] * ria[1];
            Sa[1] += ria[0] * ria[1];
            Sa[2] += ria[1] * ria[0];
        }
        
        S.push_back(Sa);
    }
    return S;
}

/**
 * @brief Calculates the anisotropy of each cell.
 */
std::vector<double> shape_anisotropy(const Regions& regions, const PointRegion& pregions, 
                                     const Vertices& vertices, const Vertices& coords) {
    auto Sa1 = shape_tensor(regions, pregions, vertices, coords);
    std::vector<double> shape_anisotropy;

    for (const auto& Sa : Sa1) {
        Matrix2d tensor;
        tensor << Sa[0], Sa[1], Sa[2], Sa[3];
        
        SelfAdjointEigenSolver<Matrix2d> eig_solver(tensor);
        Vector2d eigenvalues = eig_solver.eigenvalues();

        double aniso = 0.0;
        if (std::abs(eigenvalues[0]) >= std::abs(eigenvalues[1])) {
            aniso = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1]);
        } else {
            aniso = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[0] + eigenvalues[1]);
        }
        shape_anisotropy.push_back(aniso);
    }
    return shape_anisotropy;
}

/**
 * @brief Calculates the principal axis of polarity for each cell.
 */
std::vector<double> cell_polarity(const Regions& regions, const PointRegion& pregions, 
                                  const Vertices& vertices, const Vertices& coords) {
    auto Sa1 = shape_tensor(regions, pregions, vertices, coords);
    std::vector<double> polarity_angles;

    for (const auto& Sa : Sa1) {
        Matrix2d tensor;
        tensor << Sa[0], Sa[1], Sa[2], Sa[3];
        
        SelfAdjointEigenSolver<Matrix2d> eig_solver(tensor);
        Matrix2d eigenvectors = eig_solver.eigenvectors();
        
        double angle = atan2(eigenvectors(1, 1), eigenvectors(0, 1)); // Polar angle from principal axis
        polarity_angles.push_back(angle);
    }
    return polarity_angles;
}
