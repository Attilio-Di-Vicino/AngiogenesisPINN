import torch
import matplotlib.pyplot as plt
from logger import Logger

logger = Logger()

def plot_results(model, device):
    logger.info("Plotting...")
    # Create test points
    nx, nt = 100, 100
    x = torch.linspace(0, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_test = X.reshape(-1, 1)
    t_test = T.reshape(-1, 1)
    
    # Get predictions
    with torch.no_grad():
        y_pred = model(x_test, t_test)
    
    # Reshape predictions
    C = y_pred[:, 0].reshape(nx, nt).cpu().numpy()
    P = y_pred[:, 1].reshape(nx, nt).cpu().numpy()
    I = y_pred[:, 2].reshape(nx, nt).cpu().numpy()
    F = y_pred[:, 3].reshape(nx, nt).cpu().numpy()

    # Plot results
    fig = plt.figure(figsize=(20, 15))
    
    # Plot C (Endothelial cells)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X.cpu().numpy(), T.cpu().numpy(), C, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Endothelial Cells (C)')
    
    # Plot P (Proteases)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X.cpu().numpy(), T.cpu().numpy(), P, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Proteases (P)')
    
    # Plot I (Inhibitors)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(X.cpu().numpy(), T.cpu().numpy(), I, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('Inhibitors (I)')
    
    # Plot F (ECM)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_surface(X.cpu().numpy(), T.cpu().numpy(), F, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('ECM (F)')

    plt.tight_layout()
    plt.show()