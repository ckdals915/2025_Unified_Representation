'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation VGG11 for CIFAR100 Dataset
******************************************************************************
'''

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import VGG11AsGNNCIFAR100, VGG11CIFAR100
from utils import save_log
from utils import train_one_epoch, evaluate
from utils import get_dataloaders_CIFAR100


def main(SEED):
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs       = 100
    dataset_name = 'CIFAR100_VGG11'
    log_dir      = 'log'

    # Experiment configurations
    experiments = [
        ('VGG11CIFAR100',       lambda: VGG11CIFAR100()),
        ('VGGAsGNNCIFAR100',    lambda: VGG11AsGNNCIFAR100()),
    ]
    
    # Batch sizes for training
    batch_sizes = [16]

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    for model_name, model_fn in experiments:
        for bs in batch_sizes:
            print(f"\n*** Training {model_name}  bs={bs} ***\n")
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            
            # Log Name
            log_name = f"SEED{SEED}_{model_name}_bs{bs}"

            # Dataloaders
            train_loader, test_loader = get_dataloaders_CIFAR100(bs, num_workers=8)
            
            # Model, criterion, and optimizer
            model     = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(),
                                  lr=0.01, momentum=0.9, weight_decay=5e-4)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer, milestones=[60,120,160], gamma=0.2)

            # Best accuracy tracker
            best_acc = 0.0
            for epoch in range(1, epochs+1):
                print(f"Epoch {epoch}/{epochs}")
                torch.manual_seed(SEED + epoch)
                torch.cuda.manual_seed_all(SEED + epoch)
                random.seed(SEED + epoch)
                np.random.seed(SEED + epoch)
                
                # Training
                t0 = time.time()
                train_loss, train_acc = train_one_epoch(train_loader, 
                                                        model, 
                                                        criterion, 
                                                        optimizer, 
                                                        device)
                t1 = time.time()
                train_elapsed = t1 - t0
                th, trem = divmod(train_elapsed, 3600)
                tm, ts = divmod(trem, 60)
                train_time_str = f"{int(th):02d}:{int(tm):02d}:{int(ts):02d}"

                # Test
                t0 = time.time()
                val_loss, val_acc = evaluate(test_loader, 
                                             model, 
                                             criterion, 
                                             device)
                t1 = time.time()
                test_elapsed = t1 - t0
                th, trem = divmod(test_elapsed, 3600)
                tm, ts = divmod(trem, 60)
                test_time_str = f"{int(th):02d}:{int(tm):02d}:{int(ts):02d}"
                
                # Scheduler step
                scheduler.step()

                # Print epoch results
                print(f"Epoch {epoch} train time: {train_time_str}, test time: {test_time_str}\n")

                # Model saving
                if val_acc > best_acc:
                    best_acc = val_acc
                    ckpt = f"checkpoint/SEED{SEED}_{model_name.lower()}_bs{bs}_best.pth"
                    torch.save(model.state_dict(), ckpt)

                # Save log
                save_log(log_dir, log_name, dataset_name, bs,
                         epoch, train_loss, train_acc,
                         val_loss,   val_acc,   train_time_str, test_time_str)

            print(f"Finished {model_name} bs={bs}. Best Acc: {best_acc:.4f}\n")


if __name__ == "__main__":
    for SEED in range(10):
        main(SEED)
