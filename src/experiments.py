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
                                 save_model=False, save_dir='models/experiments'):
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
        early_stopping=True,
        patience=15,
        min_delta=0.001
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


def save_experiment_results(results_dict, filename='results/experiments.json'):
    """
    Speichert Experiment-Ergebnisse als JSON.
    
    Args:
        results_dict: Dictionary mit Experiment-Ergebnissen
        filename: Dateiname zum Speichern
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Konvertiere numpy arrays zu Listen für JSON
    json_results = {}
    for key, value in results_dict.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (list, np.ndarray)):
                    json_results[key][sub_key] = [float(x) if isinstance(x, (np.floating, float)) else x 
                                                  for x in sub_value]
                else:
                    json_results[key][sub_key] = sub_value
        elif isinstance(value, (list, np.ndarray)):
            json_results[key] = [float(x) if isinstance(x, (np.floating, float)) else x 
                                for x in value]
        else:
            json_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Ergebnisse gespeichert: {filename}")
