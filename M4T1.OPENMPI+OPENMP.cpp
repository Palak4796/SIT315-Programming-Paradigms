#include <mpi.h>          // MPI for distributed computing
#include <omp.h>          // OpenMP for multithreading
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <thread>
#include <chrono>
#include <sys/resource.h> // For checking memory usage

// Structure representing an agent (civilian or rescue worker)
struct Agent {
    int id;
    int x, y;
    bool is_rescue;
    bool is_alive;
};

// Constants for simulation
const int GRID_SIZE = 80;
const int AGENT_COUNT = 100000;
const int TIME_STEPS = 30;

// Function to print memory usage
void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory Usage (in kilobytes): " << usage.ru_maxrss << std::endl;
}

// Check if a cell is a disaster zone during this step
bool is_disaster_cell(int x, int y, int step) {
    return (y == GRID_SIZE / 2 && x >= (step % GRID_SIZE) && x <= (step % GRID_SIZE + 3));
}

// Move agent toward a target location
void move_toward(Agent& a, int tx, int ty) {
    if (a.x < tx) a.x++;
    else if (a.x > tx) a.x--;
    if (a.y < ty) a.y++;
    else if (a.y > ty) a.y--;
}

// Write agent states to a CSV file
void write_csv(const std::vector<Agent>& agents, int step, int rank) {
    std::filesystem::create_directory("mpi_logs");
    std::ostringstream fname;
    fname << "mpi_logs/step_" << step << "_rank_" << rank << ".csv";
    std::ofstream file(fname.str());
    file << "id,x,y,is_rescue,is_alive\n";
    for (const auto& a : agents) {
        file << a.id << "," << a.x << "," << a.y << "," << a.is_rescue << "," << a.is_alive << "\n";
    }
    file.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    srand(time(0) + rank);  // Seed random differently on each rank

    double start_time = MPI_Wtime();  // Record start time

    std::vector<Agent> agents;
    // Initialize agents with random positions
    for (int i = 0; i < AGENT_COUNT; ++i) {
        bool is_rescue = (i % 4 == 0);  // 1 in 4 agents are rescue agents
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        agents.push_back({i, x, y, is_rescue, true});
    }

    print_memory_usage();  // Print memory usage before simulation

    // Simulation steps
    for (int step = 0; step < TIME_STEPS; ++step) {
        if (rank == 0) std::cout << "========== Step " << step << " ==========" << std::endl;

        int distress[2] = {-1, -1};  // Location of first alive civilian
        int alive_civilians = 0;

        // Move civilians randomly and check if they enter disaster zones
        #pragma omp parallel for reduction(+:alive_civilians)
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (!a.is_alive) continue;

            if (!a.is_rescue) {
                int dx = rand() % 3 - 1;
                int dy = rand() % 3 - 1;
                a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));

                if (is_disaster_cell(a.x, a.y, step)) {
                    a.is_alive = false;  // Civilian dies in disaster
                } else {
                    alive_civilians++;  // Count alive civilians
                    #pragma omp critical
                    {
                        if (distress[0] == -1) {
                            distress[0] = a.x;
                            distress[1] = a.y;  // Set distress location
                        }
                    }
                }
            }
        }

        // Share distress location with all ranks
        MPI_Bcast(distress, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0 && distress[0] >= 0) {
            std::cout << "[Rank " << rank << "] Broadcast distress at (" << distress[0] << "," << distress[1] << ")" << std::endl;
        }
        std::cout << "[Rank " << rank << "] Alive civilians = " << alive_civilians << std::endl;

        // Move rescue agents toward distress or randomly if none
        #pragma omp parallel for
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (a.is_rescue && a.is_alive) {
                if (distress[0] >= 0) {
                    move_toward(a, distress[0], distress[1]);
                } else {
                    int dx = rand() % 3 - 1;
                    int dy = rand() % 3 - 1;
                    a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                    a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));
                }
            }
        }

        int rescued = 0;
        // Mark civilians near distress as rescued
        #pragma omp parallel for reduction(+:rescued)
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (!a.is_rescue && a.is_alive && distress[0] >= 0 &&
                std::abs(a.x - distress[0]) <= 1 && std::abs(a.y - distress[1]) <= 1) {
                a.is_alive = false;
                rescued++;
            }
        }

        std::cout << "[Rank " << rank << "] Rescued this step: " << rescued << std::endl;

        // Reduce rescued counts from all ranks to get total
        int total = 0;
        MPI_Reduce(&rescued, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Total rescued at step " << step << ": " << total << std::endl;
        }

        // Save current state to CSV
        write_csv(agents, step, rank);

        MPI_Barrier(MPI_COMM_WORLD);  // Sync all ranks
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // Slow down steps
    }

    double end_time = MPI_Wtime();  // Record end time
    if (rank == 0) {
        std::cout << "Total execution time: " << end_time - start_time << " seconds." << std::endl;
    }

    print_memory_usage();  // Print memory usage after simulation

    MPI_Finalize();  // Clean up MPI
    return 0;
}
