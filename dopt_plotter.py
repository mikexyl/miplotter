from posixpath import dirname
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import logging
import os
import sys
import yaml
import argparse

from numpy._core.fromnumeric import size

from plotting import graph

logger = logging.getLogger(__name__)

draw_outliers = True
num_robots = 5
plotting_keys = ["iSAM2Update_", "get_beliefs_", "VPCM_"]
plotting_lines = []
for key in plotting_keys:
    for i in range(num_robots):
        plotting_lines.append(key + str(i))

labels = {
    # "marginalCovariance": "Marginalization",
    # "updateConsistentSets": "Vertical PCM",
    "iSAM2Update_": "iSAM2 Update",
    "get_beliefs_": "Get Beliefs",
    "VPCM_": "VPCM",
}


def unzip_file(zip_file):
    # change the directory to the directory of the zip file
    dest_dir = zip_file.replace(".zip", "")
    if not os.path.exists(dest_dir):
        print("unzipping file")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dirname(zip_file))
        return
    print("file already unzipped")
    print(dest_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot the DOPT results")
    parser.add_argument(
        "data", type=str, help="The zip file containing the DOPT results"
    )
    args = parser.parse_args()

    datafile = "data/" + args.data
    if datafile.endswith(".zip"):
        unzip_file(datafile)
        datafolder = datafile.replace(".zip", "")
    else:
        datafolder = datafile

    # get the list of files in the data folder
    files = os.listdir(datafolder)
    # there should be a logs and a graphs folder
    logs = [f for f in files if "logs" in f]
    graphs = [f for f in files if "graphs" in f]
    assert len(logs) == 1
    assert len(graphs) == 1

    # read dataset name from noisy folder
    input_data_dir = datafolder + "/noisy"
    dataset_name = None
    # find the file ends with .g2o or .graph
    for file in os.listdir(input_data_dir):
        if file.endswith(".g2o") or file.endswith(".graph"):
            dataset_name = file.split(".")[0]
            break

    # get the list of files in the graphs folder
    files = os.listdir(datafolder + "/" + graphs[0])
    # there should be a ground_truth.csv file
    assert "ground_truth.csv" in files
    files.remove("ground_truth.csv")

    # split files based on the second last part
    agent_files = {}
    for f in files:
        agent = f.split("_")[-2]
        if agent not in agent_files:
            agent_files[agent] = []
        agent_files[agent].append(f)

    for files in agent_files.values():
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    ground_truth_file = datafolder + "/GT.txt"
    [gt_keys, gt_graph] = graph.loadTxt(ground_truth_file)
    print("Ground truth keys:", len(gt_keys))
    print("Ground truth graph:", gt_graph.shape)
    print(gt_graph)
    last_graphs = {}
    for i in range(num_robots):
        last_graphs[str(i)] = None

    rmse_data = {}
    for agent_id, files in agent_files.items():
        rmse_data[agent_id] = []
        for f in files:
            if f == "ground_truth.csv":
                continue

            print("Comparing", f, "with ground_truth.csv")

            filepath = datafolder + "/" + graphs[0] + "/" + f
            # load the csv file
            [keys, traj] = graph.loadCSV(filepath)
            if len(traj) == 0:
                continue
            aligned_traj, _, _, rmse = graph.align_trajectories(
                gt_graph, traj, gt_keys, keys
            )
            rmse_data[agent_id].append(rmse)
            if last_graphs[agent_id] is None:
                last_graphs[agent_id] = [keys, traj]

    # find the *_info.log file in the logs folder
    # and print the content
    log_file = (
        datafolder
        + "/"
        + logs[0]
        + "/"
        + [f for f in os.listdir(datafolder + "/" + logs[0]) if "info.log" in f][0]
    )

    plotting_data = {}
    for plotting_line in plotting_lines:
        plotting_data[plotting_line] = []

    with open(log_file, "r") as f:
        line = f.readline()
        # read line
        while line:
            if not line:
                logging.warning("Log file is empty")
                return

            for plot_item in plotting_lines:
                if line.startswith(plot_item):
                    # remove the plot_item from the line
                    line = line.replace(plot_item, "")
                    # get the second column, and split by space
                    data = line.split()[0]
                    # append the data to the plotting data
                    plotting_data[plot_item].append(float(data))
                    break

            line = f.readline()

    # taking average of each catorgory of data
    for key in plotting_keys:
        data = [plotting_data[key + str(i)] for i in range(num_robots)]
        data = [sum(x) / len(x) for x in zip(*data)]
        plotting_data[key] = data

    stacked_data = None
    prev_stacked_data = None
    colormap = plt.cm.viridis
    n_colors = len(labels)
    color_i = 0
    for key, label in labels.items():
        data = plotting_data[key]
        if stacked_data is None:
            stacked_data = data
        else:
            stacked_data = [sum(x) for x in zip(stacked_data, data)]
        # random color
        color = colormap(color_i / n_colors)
        # if this is the last key, plot the stacked data

        if prev_stacked_data is not None:
            plt.fill_between(
                range(0, len(stacked_data)),
                prev_stacked_data,
                stacked_data,
                alpha=0.5,
                color=color,
                label=label,
            )
        else:
            plt.fill_between(
                range(0, len(stacked_data)),
                0,
                stacked_data,
                alpha=0.5,
                color=color,
                label=label,
            )

        if key == plotting_keys[-1]:
            plt.plot(
                range(0, len(stacked_data)), stacked_data, label="Total", color="red"
            )
        prev_stacked_data = stacked_data
        color_i += 1

    plt.grid()
    # aspect ratio 3:4
    plt.gca().set_aspect(1 / 1.618, adjustable="box")
    plt.xlabel("Step")
    plt.ylabel("Time (ms)")
    plt.legend()
    # plt.title("AIR computation time breakdown.")
    plt.savefig(datafolder + "/time.pdf", format="pdf")
    plt.show(block=False)

    # new figure for trajectory
    plt.figure()
    # plot the ground truth
    plt.scatter(
        -gt_graph[:, 1], gt_graph[:, 0], label="Ground Truth", color="black", s=0.3
    )

    # draw outliers
    noise_yaml = datafolder + "/noisy/noise.yaml"
    # Read the YAML file
    with open(noise_yaml, "r") as file:
        outliers = yaml.safe_load(file)["vertices"]

    print(outliers)

    for edge in outliers:
        # find the index of the first and second vertex
        i1 = np.where(gt_keys == edge[0])[0][0]
        i2 = np.where(gt_keys == edge[1])[0][0]
        p1 = gt_graph[i1]
        p2 = gt_graph[i2]
        plt.plot(
            [-p1[1], -p2[1]],
            [p1[0], p2[0]],
            color="red",
            linewidth=2,
            alpha=0.2,
            label="Outliers",
        )

    rmse_mean = []
    for agent, key_traj in last_graphs.items():
        if key_traj is None:
            continue
        keys, traj = key_traj
        aligned_traj, _, _, rmse = graph.align_trajectories(
            gt_graph, traj, gt_keys, keys
        )
        print("RMSE", agent, rmse)
        rmse_mean.append(rmse)
        plt.scatter(
            -aligned_traj[:, 1], aligned_traj[:, 0], label=r"$R_" + agent + "$", s=0.3
        )
    print("RMSE mean", np.mean(rmse_mean))

    # set x and y to be equal
    plt.axis("equal")
    # aspect ratio 21:9
    plt.gca().set_aspect("equal", adjustable="box")
    # turn on the grid
    plt.grid(True)
    # Get current handles and labels
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    # Create a dictionary to avoid duplicates
    by_label = dict(zip(legend_labels, handles))
    # Add the legend with custom settings
    plt.legend(
        by_label.values(),
        by_label.keys(),
        markerscale=10,
        handletextpad=0.5,
        fontsize="medium",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),  # Adjusted the vertical position further down
        ncol=(num_robots + 2),
        frameon=False,  # Draw a frame around the legend
        borderpad=0.3,  # Padding inside the legend box
        labelspacing=1,  # Vertical space between legend entries
        handlelength=2,  # Length of the legend markers
        handleheight=2,  # Height of the legend markers
        borderaxespad=0.5,  # Padding between the axes and the legend box
        columnspacing=0.5,  # Space between the columns
    )
    plt.savefig(
        datafolder + "/trajectory.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,  # Padding around the figure
    )
    # plt.title(f"Trajectories of {num_robots} robots on {dataset_name} dataset")
    plt.show(block=False)

    # plot rmse
    fig = plt.figure()
    for agent, rmses in rmse_data.items():
        # reverse rmse
        rmses = rmses[::-1]
        plt.plot(rmses, label=agent)

    plt.xlabel("Step")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.savefig(datafolder + "/rmse.pdf", format="pdf")
    plt.show(block=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
