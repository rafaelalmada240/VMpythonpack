// tissue_dynamics.cpp
#include "tissue_dynamics.h"
#include "topology.h"
#include "geometry.h"

// Compute energy of a vertex in the absence of a wound
double energy_vtx_v1(const PointRegion& point_region, const Regions& regions,
                     const Ridges& ridges, const Vertices& vertices,
                     int vertex, double K, const Areas& A0, 
                     const std::vector<double>& G, double L,
                     const std::vector<int>& boundary_tissue) {
    auto [R, N_c] = find_vertex_neighbour_centers(regions, point_region, vertex);
    auto N_v = find_vertex_neighbour_vertices(ridges, vertex);

    double E = 0.0;

    std::vector<int> Ncw(N_c.begin(), N_c.end());
    Perimeters P = perimeters_vor(point_region, regions, vertices, ridges, Ncw);
    Areas A = areas_vor(point_region, regions, vertices, ridges, Ncw);
    
    Areas A0c;
    for (int i : Ncw) {
        A0c.push_back(A0[i]);
    }

    for (size_t i = 0; i < A.size(); ++i) {
        E += K / 2 * std::pow(A[i] - A0c[i], 2) + G[i] / 2 * std::pow(P[i], 2);
    }

    for (int j : N_v) {
        if (std::find(boundary_tissue.begin(), boundary_tissue.end(), vertex) == boundary_tissue.end()) {
            Point2D v = vertices[j];        
            Point2D edgeV = vertices[vertex] - v;
            double lj = edgeV.norm();
            E += -L * lj;
        }
    }

    return E;
}

// Calculate total energy of the tissue
double energy_vtx_total(const PointRegion& point_region, const Regions& regions,
                        const Ridges& ridges, const Vertices& vertices,
                        double K, const Areas& A0, const std::vector<double>& G, double L) {
    double E = 0.0;

    Perimeters P = perimeters_vor(point_region, regions, vertices, ridges, point_region);
    Areas A = areas_vor(point_region, regions, vertices, ridges, point_region);
    
    for (size_t i = 0; i < A.size(); ++i) {
        E += K / 2 * std::pow(A[i] - A0[i], 2) + G[i] / 2 * std::pow(P[i], 2);
    }

    for (int vertex = 0; vertex < vertices.size(); ++vertex) {
        auto N_v = find_vertex_neighbour_vertices(ridges, vertex);
        for (int j : N_v) {
            Point2D v = vertices[j];        
            Point2D edgeV = vertices[vertex] - v;
            double lj = edgeV.norm();    
            E += -L / 2 * lj;
        }
    }

    return E;
}

// Displace a vertex and return the new position
std::vector<Point2D> displacement_vertex(Vertices& vertices, int vertex, 
                                          double h, const Point2D& dir, int pos) {
    vertices[vertex] += h * dir * pos;
    return vertices;
}

// Calculate forces on a vertex using finite differences
Eigen::Vector2d force_vtx_finite_gradv2(const PointRegion& point_region, const Regions& regions,
                                         const Ridges& ridges, Vertices& vertices,
                                         int vertex, double K, const Areas& A0, 
                                         const std::vector<double>& G, double L, 
                                         double h, const std::vector<int>& boundary_tissue) {
    Eigen::Vector2d f_v = Eigen::Vector2d::Zero();
    
    Point2D n1(1, 0);
    Point2D n2(0, 1);
    
    auto new_vertices1x = displacement_vertex(vertices, vertex, h, n1, -1);
    auto new_vertices2x = displacement_vertex(vertices, vertex, h, n1, 1);
    auto new_vertices1y = displacement_vertex(vertices, vertex, h, n2, -1);
    auto new_vertices2y = displacement_vertex(vertices, vertex, h, n2, 1);
        
    double Ev1x = energy_vtx_v1(point_region, regions, ridges, new_vertices1x, vertex, K, A0, G, L, boundary_tissue);
    double Ev2x = energy_vtx_v1(point_region, regions, ridges, new_vertices2x, vertex, K, A0, G, L, boundary_tissue);
    double Ev1y = energy_vtx_v1(point_region, regions, ridges, new_vertices1y, vertex, K, A0, G, L, boundary_tissue);
    double Ev2y = energy_vtx_v1(point_region, regions, ridges, new_vertices2y, vertex, K, A0, G, L, boundary_tissue);
        
    double dEdx = 0.5 * (Ev2x - Ev1x) / h;
    double dEdy = 0.5 * (Ev2y - Ev1y) / h;
        
    f_v = -(dEdx * n1 + dEdy * n2);
    
    return f_v;
}

// Calculate elastic wound forces for vertices
std::vector<Eigen::Vector2d> force_vtx_elastic_wound(const Regions& regions, 
                                                      const PointRegion& point_region, 
                                                      const Ridges& ridges, 
                                                      const std::vector<double>& K, 
                                                      const Areas& A0, 
                                                      const std::vector<double>& G, double L, 
                                                      double Lw, const Vertices& vertices, 
                                                      const Centers& centers, 
                                                      int wloc, double h, const std::vector<int>& boundary_tissue) {
    std::vector<Eigen::Vector2d> F_V(vertices.size(), Eigen::Vector2d::Zero());
    double r0 = std::sqrt(std::accumulate(A0.begin(), A0.end(), 0.0) / (A0.size() * M_PI));

    auto boundary_wound = find_wound_boundary(regions, point_region, wloc);

    for (size_t v = 0; v < vertices.size(); ++v) {
        if (std::find(boundary_tissue.begin(), boundary_tissue.end(), v) == boundary_tissue.end()) {
            auto f_v = force_vtx_finite_gradv2(point_region, regions, ridges, vertices, v, K[0], A0, G, L, h, boundary_tissue);
            auto [NeighR, NeighC] = find_vertex_neighbour_centers(regions, point_region, v);
            for (size_t i = 0; i < NeighC.size(); ++i) {
                for (size_t j = i + 1; j < NeighC.size(); ++j) {
                    Point2D ci = centers[NeighC[i]];
                    Point2D cj = centers[NeighC[j]];
                    double rij = (cj - ci).norm();
                    Point2D nij = (cj - ci) / (rij + h);
                    if (rij <= r0) {
                        f_v += 0.1 * (rij - r0) * nij;
                    }
                }
            }
            F_V[v] = f_v;
        } else {
            F_V[v] = Eigen::Vector2d::Zero(); 
        }
    }
    
    return F_V;
}

// Calculate elastic forces on vertices
std::vector<Eigen::Vector2d> force_vtx_elastic(const Regions& regions, 
                                                const PointRegion& point_region, 
                                                const Ridges& ridges, 
                                                const std::vector<double>& K, 
                                                const Areas& A0, 
                                                const std::vector<double>& G, double L, 
                                                const Vertices& vertices, 
                                                const Centers& centers, 
                                                double h, const std::vector<int>& boundary_tissue) {
    std::vector<Eigen::Vector2d> F_V(vertices.size(), Eigen::Vector2d::Zero());

    double r0 = std::sqrt(std::accumulate(A0.begin(), A0.end(), 0.0) / (A0.size() * M_PI));

    for (size_t v = 0; v < vertices.size(); ++v) {
        if (std::find(boundary_tissue.begin(), boundary_tissue.end(), v) == boundary_tissue.end()) {
            auto f_v = force_vtx_finite_gradv1(point_region, regions, ridges, vertices, v, K[0], A0, G, L, h, boundary_tissue);
            auto [NeighR, NeighC] = find_vertex_neighbour_centers(regions, point_region, v);
            for (size_t i = 0; i < NeighC.size(); ++i) {
                for (size_t j = i + 1; j < NeighC.size(); ++j) {
                    Point2D ci = centers[NeighC[i]];
                    Point2D cj = centers[NeighC[j]];
                    double rij = (cj - ci).norm();
                    Point2D nij = (cj - ci) / (rij + h);
                    if (rij <= r0) {
                        f_v += 0.1 * (rij - r0) * nij;
                    }
                }
            }
            F_V[v] = f_v;
        } else {
            F_V[v] = Eigen::Vector2d::Zero(); 
        }
    }
    
    return F_V;
}

// Calculate stress for each cell based on vertex forces
std::vector<double> stress_cell(const Regions& regions, 
                                 const PointRegion& point_region, 
                                 const Vertices& vertices,
                                 const Centers& centers, 
                                 const std::vector<Eigen::Vector2d>& F) {
    std::vector<double> stress_values(regions.size(), 0.0);

    for (size_t r = 0; r < regions.size(); ++r) {
        double total_force = 0.0;
        for (int vertex : regions[r]) {
            total_force += F[vertex].norm();
        }
        stress_values[r] = total_force / regions[r].size();
    }

    return stress_values;
}

// Calculate average position of vertices for each cell
std::vector<Centers> average_vertex_positions(const Regions& regions, const Vertices& vertices) {
    std::vector<Centers> avg_positions(regions.size());

    for (size_t r = 0; r < regions.size(); ++r) {
        Centers& centers = avg_positions[r];
        for (int vertex : regions[r]) {
            centers.push_back(vertices[vertex]);
        }
    }

    return avg_positions;
}
