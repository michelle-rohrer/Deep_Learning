import torch
import numpy as np
from torch.utils.data import DataLoader
import wandb
import os
import json
import gc  # Garbage Collection für Speicherbereinigung
from src.test_train import train_model
from src.evaluation import evaluate_model


def run_hyperparameter_experiment(experiment_name, model_class, model_kwargs,
                                 train_loader, val_loader, test_loader=None,
                                 num_epochs=60, learning_rate=0.01, 
                                 batch_size=64, device=None, use_wandb=True,
                                 save_model=False, save_dir='models/experiments',
                                 optimizer_type='sgd', optimizer_momentum=0.0,
                                 early_stopping=False):
    """
    Führt ein Hyperparameter-Experiment durch.
    
    Args:
        experiment_name: Name des Experiments (für wandb und Dateinamen)
        model_class: Modell-Klasse (z.B. FlexibleCNN)
        model_kwargs: Dictionary mit Argumenten für Modell-Initialisierung
        train_loader: DataLoader für Training
        val_loader: DataLoader für Validation
        test_loader: DataLoader für Test (optional)
        num_epochs: Anzahl Epochen
        learning_rate: Lernrate
        batch_size: Batch-Größe
        device: Device (wird automatisch erkannt wenn None)
        use_wandb: Ob wandb logging aktiviert werden soll
        save_model: Ob Modell gespeichert werden soll
        save_dir: Verzeichnis zum Speichern
        optimizer_type: Typ des Optimizers ('sgd' oder 'adam')
        optimizer_momentum: Momentum für SGD (nur bei optimizer_type='sgd')
        early_stopping: Ob Early Stopping aktiviert werden soll (Standard: False für fairen Vergleich)
    
    Returns:
        dict: Ergebnisse mit allen Metriken
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() 
                             else "cuda" if torch.cuda.is_available() 
                             else "cpu")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Modell-Parameter: {model_kwargs}")
    print(f"Device: {device}")
    print(f"Epochen: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    
    # Modell erstellen
    model = model_class(**model_kwargs).to(device)
    
    # Parameter zählen
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Modell-Parameter: {total_params:,} total, {trainable_params:,} trainable")
    
    # Training
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        run_name=experiment_name,
        early_stopping=early_stopping,  # Wird als Parameter übergeben (Standard: False für fairen Vergleich)
        patience=15,
        min_delta=0.001,
        optimizer_type=optimizer_type,
        optimizer_momentum=optimizer_momentum
    )
    
    # Beste Performance finden
    best_epoch_idx = np.argmax(val_accs)
    best_val_acc = val_accs[best_epoch_idx]
    best_val_f1 = val_f1s[best_epoch_idx]
    
    # Evaluation auf Test-Set (falls vorhanden)
    test_results = None
    if test_loader is not None:
        test_results = evaluate_model(model, device, test_loader, 
                                     class_names=None, use_wandb=False)
    
    # Ergebnisse zusammenstellen
    results = {
        'experiment_name': experiment_name,
        'model_kwargs': model_kwargs,
        'num_params': total_params,
        'num_trainable_params': trainable_params,
        'num_epochs_trained': len(train_accs),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        
        # Trainingskurven
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        
        # Finale Metriken
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_f1': train_f1s[-1],
        'final_val_f1': val_f1s[-1],
        
        # Beste Metriken
        'best_epoch': best_epoch_idx + 1,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'best_train_acc': train_accs[best_epoch_idx],
        'best_train_f1': train_f1s[best_epoch_idx],
        
        # Overfitting-Gap
        'overfitting_gap_acc': train_accs[-1] - val_accs[-1],
        'overfitting_gap_f1': train_f1s[-1] - val_f1s[-1],
        
        # Test-Ergebnisse (falls vorhanden)
        'test_results': test_results
    }
    
    # Modell speichern (optional)
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{experiment_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Modell gespeichert: {model_path}")
    
    # Ergebnisse ausgeben
    print(f"\n{'='*60}")
    print(f"ERGEBNISSE: {experiment_name}")
    print(f"{'='*60}")
    print(f"Beste Validation Accuracy: {best_val_acc:.2f}% (Epoche {best_epoch_idx + 1})")
    print(f"Finale Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"Finale Validation F1-Score: {val_f1s[-1]:.4f}")
    print(f"Overfitting-Gap (Acc): {results['overfitting_gap_acc']:.2f}%")
    if test_results:
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"{'='*60}\n")
    
    # SPEICHERBEREINIGUNG: Explizit Modell und Cache löschen
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # MPS Cache leeren (Apple Silicon)
    gc.collect()  # Garbage Collection
    print("✓ Speicher nach Experiment freigegeben")
    
    return results


def _convert_numpy_types(obj):
    """
    Rekursive Funktion zur Konvertierung von NumPy-Typen zu Python-native Typen.
    
    Konvertiert:
    - np.integer (int64, int32, etc.) → int
    - np.floating (float64, float32, etc.) → float
    - np.ndarray → list
    - Listen und Dictionaries werden rekursiv durchlaufen
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(x) for x in obj]
    else:
        return obj


def save_experiment_results(results_dict, filename='results/experiments.json'):
    """
    Speichert Experiment-Ergebnisse als JSON.
    
    Args:
        results_dict: Dictionary mit Experiment-Ergebnissen
        filename: Dateiname zum Speichern
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Konvertiere alle NumPy-Typen zu Python-native Typen
    json_results = _convert_numpy_types(results_dict)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Ergebnisse gespeichert: {filename}")


def save_hypothesis_results(results_dict, filename):
    """
    Speichert Hypothesen-Ergebnisse in einer JSON-Datei.
    
    Args:
        results_dict: Dictionary mit den Ergebnissen
        filename: Pfad zur JSON-Datei
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    save_experiment_results(results_dict, filename)
    print(f"✓ Hypothesen-Ergebnisse gespeichert: {filename}")


def load_hypothesis_results(filename):
    """
    Lädt Hypothesen-Ergebnisse aus einer JSON-Datei.
    
    Args:
        filename: Pfad zur JSON-Datei
        
    Returns:
        dict: Geladene Ergebnisse oder None, falls Datei nicht existiert
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
        
        # Listen zurück in numpy arrays konvertieren (falls nötig)
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key in ['train_losses', 'val_losses', 'train_accs', 'val_accs', 'train_f1s', 'val_f1s']:
                    if sub_key in value and isinstance(value[sub_key], list):
                        value[sub_key] = np.array(value[sub_key])
            elif isinstance(value, list):
                results[key] = np.array(value)
        
        return results
    return None
