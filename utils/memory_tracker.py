import torch


def get_gpu_memory_stats() -> dict:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        stats = {
            'allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved(device) / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        }
        return stats
    else:
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0}


def print_gpu_memory_stats(prefix: str = ""):
    stats = get_gpu_memory_stats()
    print(f"{prefix}GPU Memory - Allocated: {stats['allocated']:.2f} GB, Cached: {stats['cached']:.2f} GB, Max Allocated: {stats['max_allocated']:.2f} GB")


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
