#include "t1_transitions.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>

// Function to calculate line coefficients (used for testing conditions)
std::pair<double, double> lineCoeffs(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
    double a = (p2.y() - p1.y()) / (p2.x() - p1.x());
    double b = p1.y() - a * p1.x();
    return {a, b};
}

// Function to rotate vertices during a T1 transition
Eigen::Vector2d T1Transitions::T1Rotations(const Eigen::Vector2d& orig, const Eigen::Vector2d& min) {
    Eigen::Matrix2d R_p, R_m;
    R_p << 0, -1, 1, 0; // Rotation matrix for 90 degrees counter-clockwise
    R_m << 0, 1, -1, 0; // Rotation matrix for 90 degrees clockwise

    Eigen::Vector2d diff = orig - min;
    double sqrt3_over_2 = std::sqrt(3) / 2.0;

    Eigen::Vector2d new_diff_1 = min + sqrt3_over_2 * (R_p * diff + 0.5 * diff);
    Eigen::Vector2d new_diff_2 = min + sqrt3_over_2 * (R_m * diff + 0.5 * diff);

    // Randomly assign the new vertices
    std::random_device rd;  // Random number generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double p = dis(gen);

    return (p > 0.5) ? new_diff_1 : new_diff_2;
}

// Function to update edges after a T1 transition
std::vector<std::vector<int>> T1Transitions::T1ChangeEdges(const std::vector<std::vector<int>>& ridges,
                                                            const std::vector<int>& vertices_neigh,
                                                            const std::vector<int>& vertices_neigh_min,
                                                            int i, 
                                                            int v_min) {
    std::vector<std::vector<int>> new_ridges = ridges;
    auto loc_ridges = findRidgeIndices(ridges, i); // Assumes this function finds indices of ridges containing 'i'
    auto loc_neigh_not_vm = findIndicesNotEqual(vertices_neigh, v_min);

    for (size_t j = 0; j < loc_ridges.size(); ++j) {
        if (v_min == new_ridges[loc_ridges[j]][0] || v_min == new_ridges[loc_ridges[j]][1]) {
            continue; // Skip if the ridge is connected to v_min
        }
        new_ridges[loc_ridges[j]] = {vertices_neigh[loc_neigh_not_vm[j]], i}; // Update ridge
    }

    loc_ridges = findRidgeIndices(ridges, v_min);
    auto loc_neigh_not_i = findIndicesNotEqual(vertices_neigh_min, i);

    for (size_t j = 0; j < loc_ridges.size(); ++j) {
        if (i == new_ridges[loc_ridges[j]][0] || i == new_ridges[loc_ridges[j]][1]) {
            continue; // Skip if the ridge is connected to i
        }
        new_ridges[loc_ridges[j]] = {vertices_neigh_min[loc_neigh_not_i[j]], v_min}; // Update ridge
    }

    return new_ridges;
}

// Main function to execute T1 transitions
std::tuple<std::vector<std::vector<int>>, std::vector<Eigen::Vector2d>, std::vector<std::vector<int>>, int> 
T1Transitions::T1Transition2(std::vector<Eigen::Vector2d>& vertices, 
                              std::vector<std::vector<int>>& ridges, 
                              std::vector<std::vector<int>>& regions, 
                              const std::vector<int>& point_region, 
                              double thresh_len) {
    int transition_counter = 0;
    std::vector<int> vertex_list(vertices.size());
    std::iota(vertex_list.begin(), vertex_list.end(), 0); // Fill with 0, 1, ..., n

    std::random_device rd;  // Random number generator
    std::mt19937 gen(rd());
    std::shuffle(vertex_list.begin(), vertex_list.end(), gen); // Shuffle vertex list

    for (int i : vertex_list) {
        if (std::isnan(vertices[i].x()) || std::isnan(vertices[i].y())) {
            continue; // Skip if vertex is NaN
        }

        // Find neighbouring vertices
        auto vertices_neigh = findVertexNeighbourVertices(ridges, i);
        std::vector<double> list_len;
        std::vector<int> list_neigh_v_not_excluded;

        for (int v : vertices_neigh) {
            if (std::find(vertex_list.begin(), vertex_list.end(), v) != vertex_list.end()) {
                list_neigh_v_not_excluded.push_back(v);
                Eigen::Vector2d deltax = vertices[i] - vertices[v];
                list_len.push_back(deltax.norm());
            }
        }

        if (list_neigh_v_not_excluded.size() > 2) {
            // Find closest neighbouring vertex
            auto loc_v_min = std::min_element(list_len.begin(), list_len.end());
            int v_min = list_neigh_v_not_excluded[std::distance(list_len.begin(), loc_v_min)];

            // Find neighbours of closest vertex
            auto vertices_neigh_min = findVertexNeighbourVertices(ridges, v_min);

            // Ensure conditions for T1 transition are met
            if (vertices_neigh_min.size() > 2 && 
                std::none_of(vertices_neigh_min.begin(), vertices_neigh_min.end(),
                             [&](int v) { return std::find(list_neigh_v_not_excluded.begin(), 
                                                            list_neigh_v_not_excluded.end(), v) != list_neigh_v_not_excluded.end(); })) {
                
                auto regions_neigh_v = findVertexNeighbourCenters(regions, point_region, i);
                auto regions_neigh_vmin = findVertexNeighbourCenters(regions, point_region, v_min);
                
                auto region_common = findCommonRegions(regions_neigh_v, regions_neigh_vmin);
                auto region_exc_v = findDifferentRegions(regions_neigh_v, region_common);
                auto region_exc_vmin = findDifferentRegions(regions_neigh_vmin, region_common);

                if (!region_exc_v.empty() && !region_exc_vmin.empty() && region_common.size() > 1 && *loc_v_min < thresh_len) {
                    // Perform the T1 transition
                    auto new_region_common = mergeRegions(region_exc_v, region_exc_vmin);
                    int i_v = (std::rand() % 2);
                    int i_min = 1 - i_v;

                    int new_region_exc_v = region_common[i_v];
                    int new_region_exc_min = region_common[i_min];

                    // Update regions for vertex i and v_min
                    regions[region_exc_vmin[0]].push_back(i);
                    regions[region_exc_v[0]].push_back(v_min);
                    removeElementFromRegion(regions[new_region_exc_v], v_min);
                    removeElementFromRegion(regions[new_region_exc_min], i);

                    // Update ridges
                    ridges = T1ChangeEdges(ridges, vertices_neigh, vertices_neigh_min, i, v_min);
                    std::tie(vertices[i], vertices[v_min]) = T1Rotations(vertices[i], vertices[v_min]);
                    transition_counter++;
                }
            }
        }
    }
    
    return std::make_tuple(ridges, vertices, regions, transition_counter);
}
