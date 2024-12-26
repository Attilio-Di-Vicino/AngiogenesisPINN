import os
import torch
import matplotlib.pyplot as plt
from logger import Logger

logger = Logger()

def single_plot(fig, n, projection, X, T, Z, cmap, name):
    # Plot C, P, I, F (Endothelial cells)
    ax1 = fig.add_subplot(n, projection=projection)
    ax1.plot_surface(X, T, Z, cmap=cmap)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title(name)


def plot_results(X, T, C, P, I, F, nx, nt):
    logger.info("Plotting...")

    # Reshape X_test e T_test di nuovo come X e T
    X = X.reshape(nx, nt)
    T = T.reshape(nx, nt)
    X = X.cpu().numpy()
    T = T.cpu().numpy()

    # Plot results
    fig = plt.figure(figsize=(20, 15))

    # Plot C (Endothelial cells)
    single_plot(fig, 221, '3d', X, T, C, 'viridis', 'Endothelial Cells (C)')
    # Plot P (Proteases)
    single_plot(fig, 222, '3d', X, T, P, 'viridis', 'Proteases (P)')
    # Plot I (Inhibitors)
    single_plot(fig, 223, '3d', X, T, I, 'viridis', 'Inhibitors (I)')
    # Plot F (ECM)
    single_plot(fig, 224, '3d', X, T, F, 'viridis', 'ECM (F)')

    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(folder_name, "plot.png"))

    return fig

def plot_loss(list_loss):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(list_loss, label='Loss', color='blue', marker='o', linestyle='-')
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
    return fig