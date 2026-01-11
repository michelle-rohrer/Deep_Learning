"""
Konfigurationsdatei für Multi-Core Optimierung auf Apple Silicon (M5)
"""

import os
import multiprocessing
import torch

def get_optimal_num_workers():
    """
    Berechnet optimale Anzahl Worker für DataLoader basierend auf CPU-Cores.
    Für M5: Nutzt alle verfügbaren Cores für Multi-Processing.
    
    Returns:
        int: Optimale Anzahl Worker (empfohlen: Anzahl Cores - 1)
    """
    num_cores = multiprocessing.cpu_count()
    # Für DataLoader: Anzahl Cores - 1 (einer bleibt für Hauptprozess)
    optimal_workers = max(1, num_cores - 1)
    return optimal_workers

def get_device():
    """
    Gibt das optimale Device für Apple Silicon zurück.
    Priorität: MPS (Metal) > CPU
    
    Returns:
        torch.device: Das optimale Device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def print_system_info():
    """
    Gibt System-Informationen aus (nützlich für Debugging).
    """
    print("=" * 60)
    print("System-Informationen für Multi-Core Optimierung")
    print("=" * 60)
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"Optimale num_workers für DataLoader: {get_optimal_num_workers()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS verfügbar: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS gebaut: {torch.backends.mps.is_built()}")
    print(f"Verwendetes Device: {get_device()}")
    print("=" * 60)

if __name__ == "__main__":
    print_system_info()
