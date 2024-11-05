// tissue_dynamics.h
#ifndef TISSUE_DYNAMICS_H
#define TISSUE_DYNAMICS_H

#include "tissue_types.h"
#include <Eigen/Dense>
#include <vector>

// Function to compute the energy of a vertex based on its neighbors
double energy_vtx_v1(const PointRegion& point_region, const Regions& regions,
                     const Ridges& ridges, const Vertices& vertices,
                     int vertex, double K, const Areas& A0, 
                     const std::vector<double>& G, double L,
                     const std::vector<int>& boundary_tissue);

// Calculate total energy for all vertices
double energy_vtx_total(const PointRegion& point_region, const Regions& regions,
                        const Ridges& ridges, const Vertices& vertices,
                        double K, const Areas& A0, const std::vector<double>& G, double L);

// Function to displace a vertex and return the new position
std::vector<Point2D> displacement_vertex(Vertices& vertices, int vertex, 
                                          double h, const Point2D& dir, int pos);

// Calculate forces on a vertex using finite differences
Eigen::Vector2d force_vtx_finite_gradv2(const PointRegion& point_region, const Regions& regions,
                                         const Ridges& ridges, Vertices& vertices,
                                         int vertex, double K, const Areas& A0, 
                                         const std::vector<double>& G, double L, 
                                         double h, const std::vector<int>& boundary_tissue);

// Calculate forces on a vertex using finite differences (version 1)
Eigen::Vector2d force_vtx_finite_gradv1(const PointRegion& point_region, const Regions& regions,
                                         const Ridges& ridges, Vertices& vertices,
                                         int vertex, double K, const Areas& A0, 
                                         const std::vector<double>& G, double L, 
                                         double h, const std::vector<int>& boundary_tissue);

// Calculate elastic wound forces for vertices
std::vector<Eigen::Vector2d> force_vtx_elastic_wound(const Regions& regions, 
                                                      const PointRegion& point_region, 
                                                      const Ridges& ridges, 
                                                      const std::vector<double>& K, 
                                                      const Areas& A0, 
                                                      const std::vector<double>& G, double L, 
                                                      double Lw, const Vertices& vertices, 
                                                      const Centers& centers, 
                                                      int wloc, double h, const std::vector<int>& boundary_tissue);

// Calculate elastic forces on vertices
std::vector<Eigen::Vector2d> force_vtx_elastic(const Regions& regions, 
                                                const PointRegion& point_region, 
                                                const Ridges& ridges, 
                                                const std::vector<double>& K, 
                                                const Areas& A0, 
                                                const std::vector<double>& G, double L, 
                                                const Vertices& vertices, 
                                                const Centers& centers, 
                                                double h, const std::vector<int>& boundary_tissue);

// Calculate stress for each cell based on vertex forces
std::vector<double> stress_cell(const Regions& regions, 
                                 const PointRegion& point_region, 
                                 const Vertices& vertices,
                                 const Centers& centers, 
                                 const std::vector<Eigen::Vector2d>& F);

// Calculate average position of vertices for each cell
std::vector<Centers> average_vertex_positions(const Regions& regions, const Vertices& vertices);

#endif // TISSUE_DYNAMICS_H
