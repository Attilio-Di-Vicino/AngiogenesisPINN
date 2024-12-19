import torch
from angiogenesis_pinn import AngiogenesisPINN
from plotter import plot_results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device:{device}")

    # Initialize the model with custom parameters
    model = AngiogenesisPINN(device=device, layers=[2, 100, 100, 4], epsilon=40, 
                             learning_rate=0.001, patience=50, n_epochs=1000)

    # Train the model
    model.fit()

    # Plot the model (trained)
    plot_results(model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))