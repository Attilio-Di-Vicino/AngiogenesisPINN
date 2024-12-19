import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AngiogenesisPINN(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Physical parameters
        self.device = device
        self.dC = 0.001
        self.dP = 0.005
        self.dI = 0.005
        self.k1 = 0.1
        self.k2 = 0.1
        self.k3 = 0.1
        self.k4 = 0.1
        self.k5 = 0.05
        self.k6 = 0.05
        self.alpha1 = 0.1
        self.alpha2 = 0.1
        self.alpha3 = 0.2
        self.alpha4 = 1
        self.epsilon = 40

        # Neural network for (C, P, I, F)
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4)
        ).to(device)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def T(self, x):
        """Tumor angiogenic factor"""
        return torch.exp(-torch.pow(1.0 - x, 2) / self.epsilon)
    
    def f_terms(self, C, T):
        """Compute chemotaxis/haptotaxis coefficients"""
        fI = self.alpha2 * C
        fF = self.alpha1 * C
        fT = self.alpha3 * C / (1 + self.alpha4 * T)
        return fI, fF, fT

    def pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        # Solutions
        y = self.forward(x, t)
        C, P, I, F = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
        T = self.T(x)

        # Time derivatives
        Ct = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C).to(self.device), create_graph=True)[0]
        Pt = torch.autograd.grad(P, t, grad_outputs=torch.ones_like(P).to(self.device), create_graph=True)[0]
        It = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I).to(self.device), create_graph=True)[0]

        # Space derivatives
        Cx = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C).to(self.device), create_graph=True)[0]
        Cxx = torch.autograd.grad(Cx, x, grad_outputs=torch.ones_like(Cx).to(self.device), create_graph=True)[0]

        Px = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(P).to(self.device), create_graph=True)[0]
        Pxx = torch.autograd.grad(Px, x, grad_outputs=torch.ones_like(Px).to(self.device), create_graph=True)[0]

        Ix = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I).to(self.device), create_graph=True)[0]
        Ixx = torch.autograd.grad(Ix, x, grad_outputs=torch.ones_like(Ix).to(self.device), create_graph=True)[0]

        Fx = torch.autograd.grad(F, x, grad_outputs=torch.ones_like(F).to(self.device), create_graph=True)[0]

        Tx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T).to(self.device), create_graph=True)[0]

        # Taxis terms
        fI, fF, fT = self.f_terms(C, T)

        # Full derivatives of taxis terms
        J_I = torch.autograd.grad((fI * Ix).sum(), x, create_graph=True)[0]
        J_F = torch.autograd.grad((fF * Fx).sum(), x, create_graph=True)[0]
        J_T = torch.autograd.grad((fT * Tx).sum(), x, create_graph=True)[0]

        # PDE residuals
        res_C = Ct - (self.dC * Cxx + J_I - J_F - J_T + self.k1 * C * (1 - C))
        res_P = Pt - (self.dP * Pxx - self.k3 * P * I + self.k4 * T * C + self.k5 * T - self.k6 * P)
        res_I = It - (self.dI * Ixx - self.k3 * P * I)
        res_F = -self.k2 * P * F

        return torch.mean(res_C**2 + res_P**2 + res_I**2 + res_F**2)
    
    def boundary_loss(self, x, t):
        """No-flux boundary conditions"""
        x_b = torch.cat([torch.zeros_like(x), torch.ones_like(x)])
        t_b = torch.cat([t, t])
        x_b.requires_grad_(True)

        y_b = self.forward(x_b, t_b)
        grad = torch.autograd.grad(y_b.sum(), x_b, create_graph=True)[0]

        return torch.mean(grad**2)

    def initial_loss(self, x):
        """Initial conditions"""
        t = torch.zeros_like(x)
        y = self.forward(x, t)
        C, P, I, F = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

        # Initial conditions from equations (5) and (6)
        a = 0.1
        C0 = torch.where(x <= a, torch.ones_like(x), torch.zeros_like(x))

        epsi1 = 0.1 * torch.rand(1).item()
        epsi2 = 0.1 * torch.rand(1).item()
        epsi3 = 0.1 * torch.rand(1).item()

        P0 = epsi1 * torch.ones_like(x).to(self.device)
        I0 = epsi2 * torch.ones_like(x).to(self.device)
        F0 = epsi3 * torch.ones_like(x).to(self.device)

        return torch.mean((C - C0)**2 + (P - P0)**2 + (I - I0)**2 + (F - F0)**2)

def train_model(n_epochs=10000):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = AngiogenesisPINN(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training points
    x = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    t = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_train = X.reshape(-1, 1)
    t_train = T.reshape(-1, 1)

    best_loss = float('inf')
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pde_l = model.pde_loss(x_train, t_train)
        bc_l = model.boundary_loss(x_train, t_train)
        ic_l = model.initial_loss(x_train)

        loss = pde_l + bc_l + ic_l
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.2e}')

    return model

def save_results(X, T, C, P, I, F):
    """Save results in MATLAB format"""
    scipy.io.savemat('angiogenesis_results.mat', {
        'x': X,
        't': T,
        'C': C,
        'P': P,
        'I': I,
        'F': F
    })
    

def plot_results(model, device):
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
    X = X.cpu().numpy()
    T = T.cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot C (Endothelial cells)
    ax1 = fig.add_subplot(221, projection='3d')
    surf1 = ax1.plot_surface(X, T, C, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Endothelial Cells (C)')
    fig.colorbar(surf1, ax=ax1)

    # Plot P (Proteases)
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, T, P, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Proteases (P)')
    fig.colorbar(surf2, ax=ax2)

    # Plot I (Inhibitors)
    ax3 = fig.add_subplot(223, projection='3d')
    surf3 = ax3.plot_surface(X, T, I, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('Inhibitors (I)')
    fig.colorbar(surf3, ax=ax3)

    # Plot F (ECM)
    ax4 = fig.add_subplot(224, projection='3d')
    surf4 = ax4.plot_surface(X, T, F, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('ECM (F)')
    fig.colorbar(surf4, ax=ax4)
    plt.tight_layout()
    plt.savefig('angiogenesis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print("\nSolution Statistics:")
    print(f"Endothelial Cells (C) - Min: {C.min():.4f}, Max: {C.max():.4f}, Mean: {C.mean():.4f}")
    print(f"Proteases (P) - Min: {P.min():.4f}, Max: {P.max():.4f}, Mean: {P.mean():.4f}")
    print(f"Inhibitors (I) - Min: {I.min():.4f}, Max: {I.max():.4f}, Mean: {I.mean():.4f}")
    print(f"ECM (F) - Min: {F.min():.4f}, Max: {F.max():.4f}, Mean: {F.mean():.4f}")

    # Save numerical data
    save_results(X, T, C, P, I, F)



if __name__ == "__main__":
    model = train_model()
    # Plot and save results
    plot_results(model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))