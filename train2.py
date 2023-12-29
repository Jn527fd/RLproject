import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    print(f"Number of CUDA GPUs available: {num_gpus}")

    # Iterate through available GPUs and print their properties
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}, CUDA Compute Capability: {gpu.major}.{gpu.minor}")
else:
    print("CUDA is not available on this system.")
