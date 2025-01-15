import torch
import argparse
from angiogenesis_pinn import AngiogenesisPINN
from sklearn.model_selection import train_test_split
from plotter import plot_results, plot_loss
from logger import Logger
from utils import check_device, print_system_info, save_all

logger = Logger()

# DONE:
# 1. Input
# 3. GPU
# 4. Time
# 5. Output

# TODO:
# 2. Model evaluation ?? 

INPUT_SIZE = 50
TEST_SIZE = 0.5
nx, nt = INPUT_SIZE, INPUT_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu', optional)")
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("Try: python main.py --device cuda\n")
        raise e

    # Control the specified device or choose automatically
    device = check_device(args.device)
    logger.info(f"Final device selected: {device}")
    
    counter = print_system_info()

    # Training data
    logger.info("Loading the dataset...")
    x = torch.linspace(0, 5, INPUT_SIZE).reshape(-1, 1).to(device)
    t = torch.linspace(0, 25, INPUT_SIZE).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_train = X.reshape(-1, 1)
    T_train = T.reshape(-1, 1)

    X_test = X.reshape(-1, 1)
    T_test = T.reshape(-1, 1)

    # logger.info("Splitting the dataset...")
    # X_train, X_test, T_train, T_test = train_test_split(X_train, T_train, test_size=TEST_SIZE, random_state=42)

    logger.info("****** Shape of dataset ******")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"T_train: {T_train.shape}")
    logger.info(f"X_test:  {X_test.shape}")
    logger.info(f"T_test:  {T_test.shape}")

    # Initialize the model with custom parameters
    model = AngiogenesisPINN(device=device, X_train=X_train, X_test=X_test, T_train=T_train, T_test=T_test, 
                             layers=[2, 100, 150, 150, 100, 4], epsilon=40, learning_rate=0.001, patience=50, n_epochs=1000)

    # Train the model
    training_time, loss = model.fit()

    fig_loss = plot_loss(list_loss=loss)

    C, P, I, F = model.predict(X_test, T_test, nx, nt)

    # Plot the model (trained)
    fig = plot_results(X_test, T_test, C, P, I, F, nx, nt, show=False)

    save_all(C, P, I, F, fig, X_train, T_train, X_test, T_test, device, counter, str(training_time), model, fig_loss=fig_loss)

    logger.info("All operations were performed successfully")

if __name__ == "__main__":
    main()
