import os
from datetime import datetime

def save_log(log_dir, log_name, dataset_name, batch_size, epoch,
             train_loss, train_acc, val_loss, val_acc,
             train_time_str, test_time_str):         
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_name}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'a') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
        f.write(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}\n")
        f.write(f"Train Time: {train_time_str}\n")   
        f.write(f"Test  Time: {test_time_str}\n")   
        f.write('-' * 40 + '\n')