import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from early_stopping import EarlyStopping
from logger import Logger

logger = Logger()

class AngiogenesisPINN(nn.Module):
    def __init__(self, device, layers=[2, 100, 100, 100, 4], epsilon=40, learning_rate=0.001, patience=50, n_epochs=10000):
        """
        Initialize the Angiogenesis PINN (Physics Informed Neural Network).

        Parameters:
        - device: The device (CPU or GPU) on which the model will run.
        - layers: List specifying the layer sizes for the neural network.
        - epsilon: A parameter controlling the tumor angiogenic factor.
        """
        super().__init__()
        self.device = device
        self.epsilon = epsilon
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
        self.learning_rate = learning_rate
        self.patience = patience
        self.n_epochs = n_epochs

        # Define a dynamically created neural network
        layers_list = []
        for i in range(len(layers) - 1):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layers_list.append(nn.Tanh())  # Using Tanh activation
        self.net = nn.Sequential(*layers_list).to(device)
        logger.info("Model created successfully")

    def forward(self, x, t):
        """Perform a forward pass through the network."""
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def T(self, x):
        """Tumor angiogenic factor (T)."""
        return torch.exp(-torch.pow(1.0 - x, 2) / self.epsilon)

    def f_terms(self, C, T):
        """Compute chemotaxis/haptotaxis coefficients."""
        fI = self.alpha2 * C
        fF = self.alpha1 * C
        fT = self.alpha3 * C / (1 + self.alpha4 * T)
        return fI, fF, fT

    def pde_loss(self, x, t):
        """Compute the PDE loss term."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        # Solutions
        y = self.forward(x, t)
        C, P, I, F = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
        T = self.T(x)

        # Time and space derivatives using autograd
        Ct = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C).to(self.device), create_graph=True)[0]
        Pt = torch.autograd.grad(P, t, grad_outputs=torch.ones_like(P).to(self.device), create_graph=True)[0]
        It = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I).to(self.device), create_graph=True)[0]

        Cx = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C).to(self.device), create_graph=True)[0]
        Cxx = torch.autograd.grad(Cx, x, grad_outputs=torch.ones_like(Cx).to(self.device), create_graph=True)[0]
        Px = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(P).to(self.device), create_graph=True)[0]
        Pxx = torch.autograd.grad(Px, x, grad_outputs=torch.ones_like(Px).to(self.device), create_graph=True)[0]
        Ix = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I).to(self.device), create_graph=True)[0]
        Ixx = torch.autograd.grad(Ix, x, grad_outputs=torch.ones_like(Ix).to(self.device), create_graph=True)[0]

        Fx = torch.autograd.grad(F, x, grad_outputs=torch.ones_like(F).to(self.device), create_graph=True)[0]
        Tx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T).to(self.device), create_graph=True)[0]

        # Taxis terms and their derivatives
        fI, fF, fT = self.f_terms(C, T)
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
        """Boundary conditions."""
        x_b = torch.cat([torch.zeros_like(x), torch.ones_like(x)])
        t_b = torch.cat([t, t])
        x_b.requires_grad_(True)

        y_b = self.forward(x_b, t_b)
        grad = torch.autograd.grad(y_b.sum(), x_b, create_graph=True)[0]

        return torch.mean(grad**2)

    def initial_loss(self, x):
        """Initial conditions."""
        t = torch.zeros_like(x)
        y = self.forward(x, t)
        C, P, I, F = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

        a = 0.1
        C0 = torch.where(x <= a, torch.ones_like(x), torch.zeros_like(x))
        epsi1, epsi2, epsi3 = 0.1 * torch.rand(3).to(self.device)

        P0 = epsi1 * torch.ones_like(x).to(self.device)
        I0 = epsi2 * torch.ones_like(x).to(self.device)
        F0 = epsi3 * torch.ones_like(x).to(self.device)

        return torch.mean((C - C0)**2 + (P - P0)**2 + (I - I0)**2 + (F - F0)**2)

    def fit(self):
        logger.info("Training...")
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Training data
        x = torch.linspace(0, 1, 100).reshape(-1, 1).to(self.device)
        t = torch.linspace(0, 1, 100).reshape(-1, 1).to(self.device)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        x_train = X.reshape(-1, 1)
        t_train = T.reshape(-1, 1)

        # Early stopping
        early_stopping = EarlyStopping(patience=self.patience)

        best_loss = float('inf')

        for epoch in tqdm(range(self.n_epochs), desc=f"[INFO] #epochs: {self.n_epochs:.2e}"):
            optimizer.zero_grad()

            pde_l = self.pde_loss(x_train, t_train)
            bc_l = self.boundary_loss(x_train, t_train)
            ic_l = self.initial_loss(x_train)

            loss = pde_l + bc_l + ic_l
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.net.state_dict(), 'best_model.pth')

            if early_stopping.should_stop(loss.item()):
                print(f"Early stopping at epoch {epoch}")
                break