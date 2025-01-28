import os
import numpy as np
import torch
import psutil
from logger import Logger

logger = Logger()

def check_device(device):
    # Controlla il dispositivo specificato o sceglie automaticamente
    if device:
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("GPU specified and is available. Usage: cuda")
        elif device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            logger.warning("GPU specified but not available. Fallback usage: cpu")
        elif device == "cpu":
            device = torch.device("cpu")
            logger.info("CPU specified. Usage: cpu")
        else:
            device = torch.device("cpu")
            logger.error(
                f"Device '{device}' is invalid. Please use a valid option. "
                f"Try: --device cuda or --device cpu. Defaulting to CPU."
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"No device specified. Defaulting to: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    return device

def print_system_info():
    counter = []
    # Number of physical and logical CPUs
    cpu_count_logical = psutil.cpu_count(logical=True)  # logical
    cpu_count_physical = psutil.cpu_count(logical=False)  # physical

    logger.info(f"Available logical CPUs: {cpu_count_logical}")
    logger.info(f"Available physical CPUs: {cpu_count_physical}")
    counter.append(f"Logical CPUs: {cpu_count_logical}")
    counter.append(f"Physical CPUs: {cpu_count_physical}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()  # Number of available GPUs
        logger.info(f"Number of available GPUs: {num_gpus}")
        counter.append(f"Number of available GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Memory in GB
            info = f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB"
            logger.info(info)
            counter.append(info)
    else:
        logger.info("No GPU available.")
        counter.append("No GPU available.")

    if torch.cuda.is_available():
        for i in range(num_gpus):
            gpu_mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Allocated memory in GB
            gpu_mem_cached = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Cache memory in GB
            info = f"GPU {i} - Allocated memory: {gpu_mem_allocated:.2f} GB, Cache memory: {gpu_mem_cached:.2f} GB"
            logger.info(info)
            counter.append(info)
    
    return counter

def get_next_folder_number(base_name="output/Angio", extension=".txt"):
    """
    Restituisce il nome della prossima cartella disponibile in base al nome base e all'estensione.
    E.g. Angio#1, Angio#2, Angio#3, ...
    """
    counter = 1
    folder_name = f"{base_name}#{counter}"
    
    # Ciclo fino a trovare un nome di cartella che non esiste
    while os.path.exists(folder_name):
        counter += 1
        folder_name = f"{base_name}#{counter}"
    
    return folder_name

def save_all(C, P, I, F, fig, X_train, T_train, X_test, T_test, device, counter, training_time, model, fig_loss):
    # Uso della funzione per creare una cartella progressiva
    folder_name = get_next_folder_number()

    # Crea la cartella
    os.makedirs(folder_name)

    # Salva i risultati in file .npy
    np.save(os.path.join(folder_name, "C.npy"), C)
    np.save(os.path.join(folder_name, "P.npy"), P)
    np.save(os.path.join(folder_name, "I.npy"), I)
    np.save(os.path.join(folder_name, "F.npy"), F)
    torch.save(model.net.state_dict(), os.path.join(folder_name, 'best_model.pth'))

    # Plot the model (trained)
    fig.savefig(os.path.join(folder_name, "plot.png"))
    fig_loss.savefig(os.path.join(folder_name, "loss.png"))

    # Scrivi le informazioni in un file di log
    with open(os.path.join(folder_name, "info.txt"), "w") as f:
        f.write(f"[INFO] ****** Shape of dataset ******\n")
        f.write(f"[INFO] X_train: {X_train.shape}\n")
        f.write(f"[INFO] T_train: {T_train.shape}\n")
        f.write(f"[INFO] X_test: {X_test.shape}\n")
        f.write(f"[INFO] T_test: {T_test.shape}\n")

        f.write(f"[INFO] ********* Info Device ********\n")
        f.write(f"[INFO] Device: {device}\n")
        for info in counter:
            f.write(f"[INFO] {info}\n")

        f.write(f"[INFO] Training time: {training_time}\n")
