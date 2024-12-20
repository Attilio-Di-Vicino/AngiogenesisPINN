import torch
from angiogenesis_pinn import AngiogenesisPINN
from plotter import plot_results
from logger import Logger

logger = Logger()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Initialize the model with custom parameters
    model = AngiogenesisPINN(device=device, layers=[2, 100, 150, 100, 4], epsilon=40, 
                             learning_rate=0.001, patience=50, n_epochs=1500)

    # Train the model
    model.fit()

    # Plot the model (trained)
    plot_results(model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))