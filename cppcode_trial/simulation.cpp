#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include "tissue_dynamics.h" // Equivalent to vertexmodelpack::sppvertex
#include "randomlattice.h"   // Equivalent to vertexmodelpack::randomlattice
#include "t1_transitions.h"  // Equivalent to vertexmodelpack::topologicaltransitions
#include "readTissueFiles.h" // Equivalent to vertexmodelpack::readTissueFiles
#include "geometry.h"        // Equivalent to vertexmodelpack::geomproperties

// Function to read user inputs from a file
std::vector<std::string> read_user_inputs(const std::string& filepath) {
    std::vector<std::string> list_inputs;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        list_inputs.push_back(line);
    }

    return list_inputs;
}

// Function to read tissue types from a file
std::vector<int> read_tissue_types(const std::string& filepath) {
    std::vector<int> tissue_types;
    std::ifstream file(filepath);
    int tissue;

    while (file >> tissue) {
        tissue_types.push_back(tissue);
    }

    return tissue_types;
}

// Initialize simulation parameters from input
void init_simulation_params(const std::vector<std::string>& inputs,
                            std::vector<int>& tissues, double& epsilonx, double& epsilont,
                            double& K_run, double& G_run, double& T, bool& UseT1,
                            bool& simple_output, int& woundsize, std::string& bigfoldername,
                            std::vector<double>& L_List, std::vector<double>& Lw_List) {
    std::istringstream iss(inputs[0]);
    int tissue;
    while (iss >> tissue) {
        tissues.push_back(tissue);
    }

    epsilonx = std::stod(inputs[1]);
    epsilont = std::stod(inputs[2]);
    K_run = std::stod(inputs[3]);
    G_run = std::stod(inputs[4]);

    bool list_plw = static_cast<bool>(std::stoi(inputs[5]));
    if (list_plw) {
        double p0_min = std::stod(inputs[13]);
        double p0_max = std::stod(inputs[14]);
        int p0_Nbin = std::stoi(inputs[15]);
        double lw_min = std::stod(inputs[16]);
        double lw_max = std::stod(inputs[17]);
        int lw_Nbin = std::stoi(inputs[18]);

        // Create dimensionless L_List and Lw_List
        for (int i = 0; i < p0_Nbin; ++i) {
            L_List.push_back(p0_min + (p0_max - p0_min) * i / (p0_Nbin - 1));
        }

        for (int i = 0; i < lw_Nbin; ++i) {
            Lw_List.push_back(lw_min + (lw_max - lw_min) * i / (lw_Nbin - 1));
        }
    } else {
        double p0_min = std::stod(inputs[6]);
        L_List.push_back(p0_min);
        double lw_min = std::stod(inputs[7]);
        Lw_List.push_back(lw_min);
    }

    double L_max = 5;
    double L_min = -5;
    double DeltaL = L_max - L_min;

    T = std::stod(inputs[8]);
    UseT1 = static_cast<bool>(std::stoi(inputs[9]));
    simple_output = static_cast<bool>(std::stoi(inputs[10]));
    woundsize = std::stoi(inputs[11]);
    bigfoldername = inputs[12];
}

// Main simulation function
void run_simulation(int tissue_n, const std::vector<std::string>& inputs,
                    const std::vector<double>& L_List, const std::vector<double>& Lw_List) {
    std::string foldername = bigfoldername + "/tissue" + std::to_string(tissue_n) + "/size" + std::to_string(woundsize);
    auto dataset = readTissueFiles::open_tissuefile(foldername, 0);

    auto coords = dataset["centers"];
    auto vorPointRegion = dataset["point regions"];
    auto vorRegions = dataset["regions"];
    auto vertices = dataset["vertices"];
    auto vorRidges = dataset["Edge connections"];
    auto Boundaries = dataset["boundaries"];
    auto wloc = dataset["WoundLoc"];

    int N = coords.size();
    std::cout << "Number of cells: " << N << std::endl;

    // Initialize simulation parameters
    auto av = geometry::areas_vor(vorPointRegion, vorRegions, vertices, vorRidges, vorPointRegion);
    double median_area = std::median(av);
    double r0 = std::sqrt(median_area / M_PI);
    double h = epsilonx * DeltaL / (2 * std::sqrt(N));
    std::cout << "Simulation spatial resolution: " << h << std::endl;

    double A0_run = median_area; // Assuming this is the intended behavior
    double mu = 1.0;
    double dt = (K_run * median_area) / mu * epsilont;
    std::cout << "Simulation temporal resolution: " << dt << std::endl;

    int M = static_cast<int>(T / dt);
    std::cout << "Simulation Max number of iterations: " << M << std::endl;

    double areaWound0 = geometry::area_vor(vorPointRegion, vorRegions, vertices, vorRidges, wloc);
    int total_transitions = 0;

    for (double lr : L_List) {
        for (double lw : Lw_List) {
            double Lr = lr * G_run * 2 * K_run * std::sqrt(median_area);
            double Lw = (lw - lr) * K_run * std::pow(median_area, 1.5);
            
            int i = 0;
            double areaWound = areaWound0;
            std::vector<double> periWList, areaWList, transitionsList;

            while (i < M && (areaWound >= areaWound0 / 8) && (areaWound <= 8 * areaWound0)) {
                // Compute perimeter and area of the wound
                double perimeterWound = geometry::perimeter_vor(vorPointRegion, vorRegions, coords, vorRidges, wloc);
                areaWound = geometry::area_vor(vorPointRegion, vorRegions, coords, vorRidges, wloc);

                // Compute forces
                Eigen::MatrixXd Rand_vertex = Eigen::MatrixXd::Zero(vertices.rows(), 2);
                Eigen::MatrixXd F_vertex = tissue_dynamics::force_vtx_elastic_wound(vorRegions, vorPointRegion, vorRidges, K_run, A0_run, G_run, Lr, Lw, coords, coords, wloc, h, Boundaries[0]);

                // Reflexive boundary conditions
                auto A_vertex = randomlattice::newWhere(coords + mu * F_vertex * dt, 6);
                coords += mu * (F_vertex + Rand_vertex) * dt;

                // Cell center positions are the average of the cell vertex positions
                coords = tissue_dynamics::cells_avg_vtx(vorRegions, vorPointRegion, coords, coords);

                // Do topological rearrangements
                int transition_counter = 0;
                if (UseT1) {
                    vorRidges = t1_transitions::T1transition(coords, vorRidges, vorRegions, vorPointRegion, 0.01 * r0);
                }

                i++;
                total_transitions += transition_counter;

                // Store results
                periWList.push_back(perimeterWound);
                areaWList.push_back(areaWound);
                transitionsList.push_back(transition_counter);
            }

            // Output of simulations for analysis
            if (simple_output) {
                readTissueFiles::simpleOutputTissues(foldername, {areaWound0, G_run, lr, lw, N}, {periWList, areaWList, transitionsList});
            } else {
                readTissueFiles::movieOutputTissues(foldername, {static_cast<int>(periWList.size()), lr, lw}, {coords, vorPointRegion, vorRegions, vertices, vorRidges, Boundaries, wloc});
            }
        }
    }

    // Log of simulations
    double tf = static_cast<double>(clock() - tme) / CLOCKS_PER_SEC;

    std::ofstream log(bigfoldername + "/log" + std::to_string(tissue_n) + ".txt", std::ios_base::app);
    log << "Median cell area - " << median_area << "\n";
    log << "Spatial resolution of the tissue - " << h << "\n";
    log << "Temporal resolution of the tissue - " << dt << "\n";
    log << "Simulation time (s) - " << tf << "\n";
    log << "Total number of iterations - " << i << "\n";
}

// Main execution block
int main() {
    std::string input_filepath = "inputrwh.txt";
    auto list_inputs = read_user_inputs(input_filepath);
    
    std::vector<int> tissues;
    double epsilonx, epsilont, K_run, G_run, T;
    bool UseT1, simple_output;
    int woundsize;
    std::string bigfoldername;
    std::vector<double> L_List, Lw_List;

    // Initialize simulation parameters
    init_simulation_params(list_inputs, tissues, epsilonx, epsilont, K_run, G_run, T, UseT1, simple_output, woundsize, bigfoldername, L_List, Lw_List);

    // Iterate over tissue types and run simulations
    for (int tissue_n : tissues) {
        run_simulation(tissue_n, list_inputs, L_List, Lw_List);
    }

    return 0;
}
