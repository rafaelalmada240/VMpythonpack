#ifndef T1_TRANSITIONS_H
#define T1_TRANSITIONS_H

#include <vector>
#include <Eigen/Dense>
#include <random>
#include <tuple>
#include "topology.h" // Include the necessary topology definitions

// Class to handle T1 transitions
class T1Transitions {
public:
    // Main function to perform T1 transitions
    static std::tuple<std::vector<std::vector<int>>, std::vector<Eigen::Vector2d>, std::vector<std::vector<int>>, int> T1Transition2(
        std::vector<Eigen::Vector2d>& vertices, 
        std::vector<std::vector<int>>& ridges, 
        std::vector<std::vector<int>>& regions, 
        const std::vector<int>& point_region, 
        double thresh_len
    );

private:
    // Helper functions for T1 transitions
    static Eigen::Vector2d T1Rotations(const Eigen::Vector2d& orig, const Eigen::Vector2d& min);
    static std::vector<std::vector<int>> T1ChangeEdges(const std::vector<std::vector<int>>& ridges, 
                                                        const std::vector<int>& vertices_neigh, 
                                                        const std::vector<int>& vertices_neigh_min, 
                                                        int i, 
                                                        int v_min);
};

#endif // T1_TRANSITIONS_H
