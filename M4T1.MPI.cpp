#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <sys/resource.h>  // For memory usage tracking

// Define the structure for an agent (either a rescue agent or a civilian)
struct Agent {
    int id;
    int x, y;
    bool is_rescue;
    bool is_alive;
};

// Constants for the simulation
const int GRID_SIZE = 80;
const int AGENT_COUNT = 100000;
const int TIME_STEPS = 30;

// Function to print current memory usage of the process
void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory Usage (in kilobytes): " << usage.ru_maxrss << std::endl;
}

// Function to check if a cell is part of the "disaster zone" at the current step
bool is_disaster_cell(int x, int y, int step) {
    return (y == GRID_SIZE / 2 && x >= (step % GRID_SIZE) && x <= (step % GRID_SIZE + 3));
}

// Function to move an agent one step toward a target (tx, ty)
void move_toward(Agent& a, int tx, int ty) {
    if (a.x < tx) a.x++;
    else if (a.x > tx) a.x--;
    if (a.y < ty) a.y++;
    else if (a.y > ty) a.y--;
}

// Save the agents' states to a CSV file for visualization
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
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get this process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    srand(time(0) + rank);  // Seed random differently per rank

    double start_time = MPI_Wtime();  // Start timing the simulation

    // Initialize all agents with random positions and types
    std::vector<Agent> agents;
    for (int i = 0; i < AGENT_COUNT; ++i) {
        bool is_rescue = (i % 4 == 0);  // 1 in 4 agents is a rescue agent
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        agents.push_back({i, x, y, is_rescue, true});
    }

    print_memory_usage();  // Show memory usage at the start

    // Start the main simulation loop
    for (int step = 0; step < TIME_STEPS; ++step) {
        if (rank == 0) std::cout << "========== Step " << step << " ==========" << std::endl;

        int distress[2] = {-1, -1};  // Coordinates of a civilian in distress
        int alive_civilians = 0;

        // Civilian movement and distress detection
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (!a.is_alive) continue;

            if (!a.is_rescue) {
                // Move randomly within the grid
                int dx = rand() % 3 - 1;
                int dy = rand() % 3 - 1;
                a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));

                // Check if civilian entered disaster cell
                if (is_disaster_cell(a.x, a.y, step)) {
                    a.is_alive = false;
                } else {
                    alive_civilians++;
                    if (distress[0] == -1) {
                        distress[0] = a.x;
                        distress[1] = a.y;
                    }
                }
            }
        }

        // Broadcast distress coordinates from rank 0 to all ranks
        MPI_Bcast(distress, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0 && distress[0] >= 0) {
            std::cout << "[Rank " << rank << "] Broadcast distress at (" << distress[0] << "," << distress[1] << ")" << std::endl;
        }
        std::cout << "[Rank " << rank << "] Alive civilians = " << alive_civilians << std::endl;

        // Move rescue agents toward distress location
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (a.is_rescue && a.is_alive) {
                if (distress[0] >= 0) {
                    move_toward(a, distress[0], distress[1]);
                } else {
                    // Move randomly if no distress call
                    int dx = rand() % 3 - 1;
                    int dy = rand() % 3 - 1;
                    a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                    a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));
                }
            }
        }

        int rescued = 0;
        // Mark nearby civilians as rescued
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (!a.is_rescue && a.is_alive && distress[0] >= 0 &&
                std::abs(a.x - distress[0]) <= 1 && std::abs(a.y - distress[1]) <= 1) {
                a.is_alive = false;
                rescued++;
            }
        }

        std::cout << "[Rank " << rank << "] Rescued this step: " << rescued << std::endl;

        // Aggregate total rescued from all processes
        int total = 0;
        MPI_Reduce(&rescued, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Total rescued at step " << step << ": " << total << std::endl;
        }

        write_csv(agents, step, rank);  // Save agent states for visualization

        MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all processes
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // Slow down simulation
    }

    double end_time = MPI_Wtime();  // Stop timing

    // Show total execution time
    if (rank == 0) {
        std::cout << "Total execution time: " << (end_time - start_time)*1000 << " milliseconds." << std::endl;
    }

    print_memory_usage();  // Show memory usage after simulation

    MPI_Finalize();  // Clean up MPI environment
    return 0;
}
