#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <cmath>
#include <iostream>
#include <numeric> // for std::accumulate
#include "topology.h"
#include "tissue_types.h" // Include the custom types

// Use the Vertices type for vertices
double perimeter_vor(const PointRegion& point_region, const Regions& regions, 
                     const Vertices& vertices, const Ridges& ridges, int i);

double area_vor(const PointRegion& point_region, const Regions& regions, 
                const Vertices& vertices, const Ridges& ridges, int i, bool to_print = false);

std::vector<double> perimeters_vor(const PointRegion& point_region, const Regions& regions, 
                                   const Vertices& vertices, const Ridges& ridges, const std::vector<int>& list_i);

std::vector<double> areas_vor(const PointRegion& point_region, const Regions& regions, 
                              const Vertices& vertices, const Ridges& ridges, const std::vector<int>& list_i, bool to_print = false);

std::vector<double> area_time(const std::vector<PointRegion>& pointRegion, const std::vector<Regions>& regions, 
                              const std::vector<std::vector<Vertices>>& vertices, const std::vector<Ridges>& ridges, int wloc, int NIter);

std::vector<double> perimeter_time(const std::vector<PointRegion>& pointRegion, const std::vector<Regions>& regions, 
                                   const std::vector<std::vector<Vertices>>& vertices, const std::vector<Ridges>& ridges, int wloc, int NIter);

std::pair<std::vector<int>, std::vector<double>> shape_neighbour(const Regions& regions, const PointRegion& pregions);

std::vector<std::vector<double>> shape_tensor(const Regions& regions, const PointRegion& point_region, 
                                              const Vertices& vertices, const Vertices& centers);

std::vector<double> shape_anisotropy(const Regions& regions, const PointRegion& pregions, 
                                     const Vertices& vertices, const Vertices& coords);

std::vector<double> cell_polarity(const Regions& regions, const PointRegion& pregions, 
                                  const Vertices& vertices, const Vertices& coords);

#endif // GEOMETRY_H
