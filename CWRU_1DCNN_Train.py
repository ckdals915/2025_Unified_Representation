'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation 1D-CNN for CWRU Dataset
******************************************************************************
'''
import time
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.CWRU_1DCNN import CNN1DCWRUAsGNN, CNN1DCWRU
from utils import save_log
from utils import train_one_epoch, evaluate
from utils import get_dataloaders_CWRU

def main(SEED):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.manual_seed(0)
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")

    # Hyperparameters
    num_epochs    = 1000
    batch_size    = 128
    learning_rate = 5e-5
    dataset_name  = "CWRU"           
    log_dir       = "log"            
    os.makedirs(log_dir, exist_ok=True)

    experiments = [
        ('CNN1DCWRUAsGNN',          lambda: CNN1DCWRUAsGNN().to(device)),
        ('CNN1DCWRU',               lambda: CNN1DCWRU().to(device)),

    ]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataloader Configuration
    train_loader, valid_loader = get_dataloaders_CWRU(batch_size=batch_size, num_workers=4)
    
    
    for model_name, model_fn in experiments:
        print(f"\n*** Training {model_name} ***\n")

        
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Model Configuration
        model = model_fn().to(device)
        
        # Criterion and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training and Validation Loop
        best_acc = 0.0
        log_name = f"SEED{SEED}_{model_name}_bs{batch_size}_{dataset_name}_lr{learning_rate}"

        for epoch in range(1, num_epochs + 1):
            print(f"\n===== Epoch {epoch}/{num_epochs} =====")

            torch.manual_seed(SEED + epoch)
            torch.cuda.manual_seed_all(SEED + epoch)
            random.seed(SEED + epoch)
            np.random.seed(SEED + epoch)

            # Training
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                train_loader, model, criterion, optimizer, device
            )
            t1 = time.time()
            train_elapsed = t1 - t0
            th, trem = divmod(train_elapsed, 3600)
            tm, ts = divmod(trem, 60)
            train_time_str = f"{int(th):02d}:{int(tm):02d}:{int(ts):02d}"

            # Validation
            t0 = time.time()
            val_loss, val_acc = evaluate(
                valid_loader, model, criterion, device
            )
            t1 = time.time()
            valid_elapsed = t1 - t0
            vh, vrem = divmod(valid_elapsed, 3600)
            vm, vs = divmod(vrem, 60)
            valid_time_str = f"{int(vh):02d}:{int(vm):02d}:{int(vs):02d}"

            # Results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Time: {train_time_str}")
            print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f} | Valid Time: {valid_time_str}")

            # Model Checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(
                    "checkpoint",
                    f"SEED{SEED}_{model_name}_{dataset_name}_lr{learning_rate}_best.pth"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"--> New best model saved (Valid Acc: {best_acc:.4f}) at {ckpt_path}")

            # Logging
            save_log(
                     log_dir=log_dir,
                     log_name=log_name,
                     dataset_name=dataset_name,
                     batch_size=batch_size,
                     epoch=epoch,
                     train_loss=train_loss,
                     train_acc=train_acc,
                     val_loss=val_loss,
                     val_acc=val_acc,
                     train_time_str=train_time_str,
                     test_time_str=valid_time_str
                    )

        print(f"\nTraining finished. Best Valid Acc: {best_acc:.4f}")


if __name__ == "__main__":
    for SEED in range(10):
        main(SEED)



