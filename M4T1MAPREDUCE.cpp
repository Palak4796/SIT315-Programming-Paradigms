#include <mpi.h>
#include <omp.h>
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
#include <sys/resource.h>  // For memory usage tracking

// Agent structure representing either a rescue agent or a civilian
struct Agent {
    int id;
    int x, y;
    bool is_rescue;
    bool is_alive;
};

// Simulation constants
const int GRID_SIZE = 80;
const int AGENT_COUNT = 100000;
const int TIME_STEPS = 50;

// Function to print memory usage
void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory Usage (in kilobytes): " << usage.ru_maxrss << std::endl;
}

// Determine if a grid cell is a disaster cell during a specific step
bool is_disaster_cell(int x, int y, int step) {
    return (y == GRID_SIZE / 2 && x >= (step % GRID_SIZE) && x <= (step % GRID_SIZE + 3));
}

// Move agent one step toward a target coordinate
void move_toward(Agent& a, int tx, int ty) {
    if (a.x < tx) a.x++;
    else if (a.x > tx) a.x--;
    if (a.y < ty) a.y++;
    else if (a.y > ty) a.y--;
}

// Write agent states to CSV file for visualization/debugging
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

// Mapper: Count civilians who entered disaster zones (treated as rescued)
int map_agents(const std::vector<Agent>& agents, int step) {
    int rescued = 0;
    for (const auto& a : agents) {
        if (!a.is_alive) continue;
        if (!a.is_rescue && is_disaster_cell(a.x, a.y, step)) {
            rescued++;
        }
    }
    return rescued;
}

// Reducer: Collect rescue counts from all MPI processes
int reduce_rescue_counts(int rank_rescue_count) {
    int total_rescues = 0;
    MPI_Reduce(&rank_rescue_count, &total_rescues, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return total_rescues;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Start MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get number of processes

    srand(time(0) + rank);  // Seed randomness differently for each process

    double start_time = MPI_Wtime();  // Start execution timer

    // Initialize agents with random positions
    std::vector<Agent> agents;
    for (int i = 0; i < AGENT_COUNT; ++i) {
        bool is_rescue = (i % 4 == 0);
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        agents.push_back({i, x, y, is_rescue, true});
    }

    print_memory_usage();  // Memory usage before simulation

    // Main simulation loop
    for (int step = 0; step < TIME_STEPS; ++step) {
        if (rank == 0) std::cout << "========== Step " << step << " ==========" << std::endl;

        int distress[2] = {-1, -1};  // Position of distress signal
        int alive_civilians = 0;

        // Civilian movement and disaster check (parallelized with OpenMP)
        #pragma omp parallel for reduction(+:alive_civilians)
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (!a.is_alive) continue;

            if (!a.is_rescue) {
                // Random movement
                int dx = rand() % 3 - 1;
                int dy = rand() % 3 - 1;
                a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));

                // Check if agent enters disaster zone
                if (is_disaster_cell(a.x, a.y, step)) {
                    a.is_alive = false;
                } else {
                    alive_civilians++;
                    // Record first distress position (critical section)
                    #pragma omp critical
                    {
                        if (distress[0] == -1) {
                            distress[0] = a.x;
                            distress[1] = a.y;
                        }
                    }
                }
            }
        }

        // Broadcast distress location from rank 0 to all processes
        MPI_Bcast(distress, 2, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0 && distress[0] >= 0) {
            std::cout << "[Rank " << rank << "] Broadcast distress at (" << distress[0] << "," << distress[1] << ")" << std::endl;
        }

        std::cout << "[Rank " << rank << "] Alive civilians = " << alive_civilians << std::endl;

        // Move rescue agents toward distress signal or wander
        #pragma omp parallel for
        for (int i = 0; i < agents.size(); ++i) {
            Agent& a = agents[i];
            if (a.is_rescue && a.is_alive) {
                if (distress[0] >= 0) {
                    move_toward(a, distress[0], distress[1]);
                } else {
                    // Wander randomly
                    int dx = rand() % 3 - 1;
                    int dy = rand() % 3 - 1;
                    a.x = std::max(0, std::min(GRID_SIZE - 1, a.x + dx));
                    a.y = std::max(0, std::min(GRID_SIZE - 1, a.y + dy));
                }
            }
        }

        // Mapper: Count locally "rescued" agents
        int local_rescue_count = map_agents(agents, step);

        // Reducer: Aggregate total rescues across all ranks
        int total_rescued = reduce_rescue_counts(local_rescue_count);

        std::cout << "[Rank " << rank << "] Rescued this step: " << local_rescue_count << std::endl;
        if (rank == 0) {
            std::cout << "Total rescued at step " << step << ": " << total_rescued << std::endl;
        }

        // Save current step data to CSV
        write_csv(agents, step, rank);

        MPI_Barrier(MPI_COMM_WORLD);  // Sync all processes
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // Pause between steps
    }

    double end_time = MPI_Wtime();  // End execution timer

    // Print execution duration
    if (rank == 0) {
        std::cout << "Total execution time: " << end_time - start_time << " seconds." << std::endl;
    }

    print_memory_usage();  // Memory usage after simulation

    MPI_Finalize();  // Clean up MPI
    return 0;
}
