import torch
import argparse
from angiogenesis_pinn import AngiogenesisPINN
from sklearn.model_selection import train_test_split
from plotter import plot_results
from logger import Logger

logger = Logger()

# DONE:
# 1. Input

# TODO:
# 2. Model evaluation ?? 
# 3. GPU
# 4. Time
# 5. Output

INPUT_SIZE = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu', optional)")
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("Try: python main.py --device cuda\n")
        raise e

    # Controlla il dispositivo specificato o sceglie automaticamente
    if args.device:
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("GPU specified and is available. Usage: cuda")
        elif args.device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            logger.warning("GPU specified but not available. Fallback usage: cpu")
        elif args.device == "cpu":
            device = torch.device("cpu")
            logger.info("CPU specified. Usage: cpu")
        else:
            device = torch.device("cpu")
            logger.error(
                f"Device '{args.device}' is invalid. Please use a valid option. "
                f"Try: --device cuda or --device cpu. Defaulting to CPU."
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"No device specified. Defaulting to: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    logger.info(f"Final device selected: {device}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Training data
    logger.info("Loading the dataset...")
    x = torch.linspace(0, 1, INPUT_SIZE).reshape(-1, 1).to(device)
    t = torch.linspace(0, 1, INPUT_SIZE).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_train = X.reshape(-1, 1)
    T_train = T.reshape(-1, 1)

    logger.info("Splitting the dataset...")
    X_train, X_test, T_train, T_test = train_test_split(X_train, T_train, test_size=0.2, random_state=42)
    logger.info("****** Shape of dataset ******")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"T_train: {T_train.shape}")
    logger.info(f"X_test:  {X_test.shape}")
    logger.info(f"T_test:  {T_test.shape}")


    # Initialize the model with custom parameters
    model = AngiogenesisPINN(device=device, X_train=X_train, X_test=X_test, T_train=T_train, T_test=T_test, 
                             layers=[2, 100, 100, 4], epsilon=40, learning_rate=0.001, patience=50, n_epochs=100)

    # Train the model
    model.fit()

    # # Plot the model (trained)
    # plot_results(model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    main()