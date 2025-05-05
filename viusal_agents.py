# visualize_agents_autoclean.py
# This script visualizes MPI simulation data and removes old PNGs before generating new ones.

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "mpi_logs/"
GRID_SIZE = 20  # Used to space out data from different MPI ranks

# Load all CSV data files, grouped by simulation step
def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "step_*_rank_*.csv")))
    all_steps = {}
    for f in files:
        base = os.path.basename(f)
        step = int(base.split("_")[1])  # Extract step number
        rank = int(base.split("_")[-1].split(".")[0])  # Extract rank number
        df = pd.read_csv(f)
        if step not in all_steps:
            all_steps[step] = []
        all_steps[step].append((rank, df))  # Group data by step and rank
    return all_steps

# Plot the agents' positions for a single simulation step
def plot_step(step_data, step_number):
    plt.figure(figsize=(10, 10))
    for rank, df in step_data:
        for _, row in df.iterrows():
            # Color code: green for live rescue, red for live civilian, gray for dead
            color = 'green' if row['is_rescue'] and row['is_alive'] else \
                    'red' if not row['is_rescue'] and row['is_alive'] else \
                    'gray'
            plt.scatter(row['x'] + rank * GRID_SIZE, row['y'], 
                        color=color, alpha=0.6, edgecolors='black', linewidths=0.3)
    plt.title(f"Agent Positions at Step {step_number}")
    plt.xlim(0, GRID_SIZE * len(step_data))  # Extend X-axis for all ranks
    plt.ylim(0, GRID_SIZE)
    plt.grid(True)
    plt.xlabel("Zone X")
    plt.ylabel("Y Position")
    plt.savefig(f"frame_step_{step_number}.png")  # Save as PNG
    plt.close()

# Delete any previously generated PNG files
def remove_old_pngs():
    old_files = glob.glob("frame_step_*.png")
    for f in old_files:
        os.remove(f)
    print("Removed old PNG files.")

# Main execution: clean old PNGs, load data, generate plots
def main():
    remove_old_pngs()
    steps = load_data()
    for step_number in sorted(steps.keys()):
        plot_step(steps[step_number], step_number)

# Entry point of the script
if __name__ == "__main__":
    main()
