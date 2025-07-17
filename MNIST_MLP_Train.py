'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation MLP for MNIST using LinearAsGNN
******************************************************************************
'''

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.MNIST_MLP import MLPAsGNNMNIST, MLPMNIST
from utils.Log_GNN import save_log
from utils.Train_Test import train_one_epoch, evaluate
from utils.DataLoader_GNN import get_dataloaders_MNIST  


def main(SEED):
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    dataset_name = 'MNIST'
    log_dir = 'log'
    experiments = [
        ('MLPAsGNNMNIST', lambda: MLPAsGNNMNIST()),
        ('MLPMNIST',       lambda: MLPMNIST())
    ]
    batch_sizes = [128]

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    for model_name, model_fn in experiments:
        for bs in batch_sizes:
            print(f"\n*** Training {model_name}  bs={bs} ***\n")

            # Set random seeds for each experiment
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            random.seed(SEED)
            np.random.seed(SEED)

            # Get data loaders, model, criterion, and optimizer
            train_loader, test_loader = get_dataloaders_MNIST(bs, SEED, num_workers=8)
            model = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-3)

            best_acc = 0.0
            log_name = f"SEED{SEED}_{model_name}_bs{bs}"

            # Train and evaluate the model
            for epoch in range(1, epochs + 1):
                print(f"Epoch {epoch}/{epochs}")

                torch.manual_seed(SEED + epoch)
                torch.cuda.manual_seed_all(SEED + epoch)
                random.seed(SEED + epoch)
                np.random.seed(SEED + epoch)


                # Train Model and Time Measurement
                t0 = time.time()
                train_loss, train_acc = train_one_epoch(
                    train_loader, model, criterion, optimizer, device)
                t1 = time.time()
                train_elapsed = t1 - t0
                th, trem = divmod(train_elapsed, 3600)
                tm, ts = divmod(trem, 60)
                train_time_str = f"{int(th):02d}:{int(tm):02d}:{int(ts):02d}"

                # Evaluate Model and Time Measurement
                t0 = time.time()
                val_loss, val_acc = evaluate(
                    test_loader, model, criterion, device)
                t1 = time.time()
                test_elapsed = t1 - t0
                th, trem = divmod(test_elapsed, 3600)
                tm, ts = divmod(trem, 60)
                test_time_str = f"{int(th):02d}:{int(tm):02d}:{int(ts):02d}"


                print(f"Epoch {epoch} train time: {train_time_str}, test time: {test_time_str}\n")

                # Print training and validation results, and save the best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    ckpt = f"checkpoint/SEED{SEED}_{model_name.lower()}_bs{bs}_best.pth"
                    torch.save(model.state_dict(), ckpt)

                # Save log
                save_log(log_dir, log_name, dataset_name, bs,
                         epoch, train_loss, train_acc,
                         val_loss, val_acc, train_time_str, test_time_str)

            print(f"Finished {model_name} bs={bs}. Best Acc: {best_acc:.4f}\n")


if __name__ == "__main__":
    for SEED in range(10):
        main(SEED)