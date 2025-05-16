import os
import sys
import random
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(
    log_file: Optional[str] = None, 
    log_level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gpu_memory_usage() -> Dict[int, int]:
    if not torch.cuda.is_available():
        return {}
    
    memory_usage = {}
    for i in range(torch.cuda.device_count()):
        memory_usage[i] = torch.cuda.memory_allocated(i) // (1024 ** 2)  # MB
    
    return memory_usage

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def print_gpu_utilization():
    import psutil
    import GPUtil
    
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        gpu_id = gpu.id
        gpu_name = gpu.name
        gpu_load = f"{gpu.load*100:.1f}%"
        gpu_free_memory = f"{gpu.memoryFree:.0f}MB"
        gpu_used_memory = f"{gpu.memoryUsed:.0f}MB"
        gpu_total_memory = f"{gpu.memoryTotal:.0f}MB"
        gpu_temperature = f"{gpu.temperature:.0f}°C"
        list_gpus.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature
        ))

    print("GPU utilization:")
    print("ID | Name | Load | Free Memory | Used Memory | Total Memory | Temperature")
    print("------------------------------------------------------------------------------------")
    for gpu_info in list_gpus:
        print(" | ".join([str(g) for g in gpu_info]))

def get_free_space_gb(path: str = '.') -> float:
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)  # Convert bytes to GB