import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_results(file, save_dir):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()

    try:
        data = pd.read_csv(file, delimiter= ',')
        s = [x.strip() for x in data.columns]
        print(s)
        x = data.values[:, 0]
        print(x)
        for i, j in enumerate([1, 2, 3, 4]):
            y = data.values[:, j]
            # y[y == 0] = np.nan  # don't show zero values
            ax[i].plot(x, y, marker='.', linewidth=2, markersize=8)
            ax[i].set_title(s[j], fontsize=12)
            # if j in [8, 9, 10]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
    except Exception as e:
        print(f'Warning: Plotting error for {file}: {e}')
    fig.savefig(save_dir + 'results.png', dpi=200)

def main():
    plot_results('results_combined.csv', '/home/schober/yolov5/runs/plots/')

if __name__ == "__main__":
    main()
