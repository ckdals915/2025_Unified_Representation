'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Components GPU Memory/Inference Time Analysis
******************************************************************************
'''
import time
import torch
import numpy as np

from models import CNN1DCWRUAsGNN, CNN1DCWRU
from models import MLPAsGNNMNIST, MLPMNIST
from models import ResNet18CIFAR100, ResNet18AsGNNCIFAR100
from models import VGG11CIFAR100, VGG11AsGNNCIFAR100
from models import ViTAsGNNCIFAR100, ViTCIFAR100


# Counting Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Formatting Parameters
def format_params(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}G"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)

# Measurement Inference time
def measure_inference_latency(model, device, input_size=(1, 1, 2048),
                              num_warmup=30, num_iters=100):
    
    # Evaluation Model
    model.eval()
    
    # Input Data
    dummy_input = torch.randn(*input_size).to(device)

    # GPU Memory Initialization
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Warming Up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Measurement
    timings = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.time()

            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.time()
            timings.append(t_end - t_start)

    # Calculate Mean, Std
    mean_time = np.mean(timings)
    std_time  = np.std(timings)

    # Calculate GPU Memory Peak
    peak_mem_bytes = 0
    if device.type == 'cuda':
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    return mean_time, std_time, peak_mem_bytes

def human_readable_bytes(n_bytes):
    # Change Units
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"

if __name__ == "__main__":
    # Device Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==> Using device: {device}\n")

    # Experiments Models
    experiments = [
        ("MLPMNIST", lambda: MLPMNIST().to(device), (1, 1, 28 * 28)),
        ("MLPAsGNNMNIST", lambda: MLPAsGNNMNIST().to(device), (1, 1, 28 * 28)),
        ("CNN1DCWRU", lambda: CNN1DCWRU().to(device), (1, 1, 2048)),
        ("CNN1DCWRUAsGNN", lambda: CNN1DCWRUAsGNN(input_length=2048).to(device), (1, 1, 2048)),
        ("VGG", lambda: VGG11CIFAR100().to(device), (1, 3, 32, 32)),
        ("VGGAsGNN", lambda: VGG11AsGNNCIFAR100().to(device), (1, 3, 32, 32)),
        ("ResNet18", lambda: ResNet18CIFAR100().to(device), (1, 3, 32, 32)),
        ("ResNet18AsGNN",  lambda: ResNet18AsGNNCIFAR100().to(device), (1, 3, 32, 32)),
        ("ViTCIFAR100", lambda: ViTCIFAR100(device=device), (1, 3, 32, 32)),
        ("ViTAsGNNCIFAR100", lambda: ViTAsGNNCIFAR100(device=device), (1, 3, 32, 32)),
    ]

    for name, ModelClass, input_size in experiments:
        print(f"===== Measuring {name} =====")
        # Model Generation
        model = ModelClass().to(device)
        model.eval()

        # Count Parameters
        total_params = count_parameters(model)
        formatted = format_params(total_params)
        print(f"  • Parameters: {formatted} ({total_params:,} 개)")

        # Measurement Inference Time & GPU Memory
        mean_time, std_time, peak_mem = measure_inference_latency(
            model, device, input_size=input_size,
            num_warmup=300, num_iters=1000
        )
        print(f"  • Inference Latency (배치=1): {mean_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        
        if device.type == 'cuda':
            print(f"  • GPU Peak Memory (Inference 중): {human_readable_bytes(peak_mem)}")
        else:
            print("  • (GPU 없음 → 메모리 측정 불가)")

        print("") 