import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb

from src.model import BaselineCNN
from src.test_train import train_model

###########################
# Evaluation der Baseline #
###########################

def evaluate_model(model, device, test_loader, class_names, use_wandb=False, run_name=None):
    """
    Detaillierte Evaluation des Modells mit allen Metriken.
    
    Args:
        model: Das zu evaluierende Modell
        device: CUDA oder CPU
        test_loader: DataLoader für Test
        class_names: Liste der Klassennamen
        use_wandb: Ob wandb logging aktiviert werden soll
        run_name: Name für den wandb run
    
    Returns:
        dict: Dictionary mit allen Metriken
    """
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Softmax für Wahrscheinlichkeiten
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Metriken berechnen
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Gewichtete Metriken
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    # Konfusionsmatrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Top-K Accuracy (Top-2)
    top2_correct = 0
    for i, (true_label, probs) in enumerate(zip(all_labels, all_probabilities)):
        top2_indices = np.argsort(probs)[-2:]
        if true_label in top2_indices:
            top2_correct += 1
    top2_accuracy = top2_correct / len(all_labels)
    
    results = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'top2_accuracy': top2_accuracy,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    # wandb logging für Evaluation
    if use_wandb:
        wandb.log({
            "test_accuracy": accuracy,
            "test_precision_weighted": precision,
            "test_recall_weighted": recall,
            "test_f1_weighted": f1,
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "test_f1_macro": f1_macro,
            "test_top2_accuracy": top2_accuracy
        })
        
        # Konfusionsmatrix als wandb Table
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_predictions,
            class_names=class_names
        )})
    
    return results


#############################
# Cross-Validation Training #
#############################
def cross_validation_training(train_dataset, model_class, num_folds=5, num_epochs=60, learning_rate=0.01, batch_size=64, img_size=48, num_classes=7):
    """
    Cross-Validation für statistische Fehlerschätzung.
    
    **WICHTIG - Konsistente Epochenanzahl:**
    Early Stopping ist deaktiviert, um sicherzustellen, dass alle Folds die gleiche Anzahl
    Epochen trainieren. Dies ist wichtig für eine faire statistische Vergleichbarkeit der Folds.
    Unterschiedliche Epochenanzahlen würden die statistische Vergleichbarkeit verfälschen.
    
    **Was wird gemacht:**
    - Teilt den Trainingsdatensatz in num_folds Folds auf
    - Trainiert für jeden Fold ein neues Modell aus der übergebenen Modell-Klasse
    - Jeder Fold trainiert exakt num_epochs Epochen (kein Early Stopping)
    - Berechnet Mittelwert und Standardabweichung über alle Folds
    
    **Interpretation der Ergebnisse:**
    - Mittelwert: Durchschnittliche Performance über alle Folds
    - Standardabweichung: Maß für die Unsicherheit/Variabilität
      - Niedrige Std: Konsistente Ergebnisse, geringe Unsicherheit
      - Hohe Std: Variable Ergebnisse, höhere Unsicherheit
      - Mögliche Ursachen für hohe Std:
        * Unterschiedliche Datenverteilungen in den Folds
        * Modell-Instabilität
        * Zu kleine Folds (wenig Daten pro Fold)
        * Modell reagiert sehr empfindlich auf Trainingsdaten
    
    **Unsicherheiten:**
    Die Standardabweichung quantifiziert die Unsicherheit in den Ergebnissen.
    Eine höhere Standardabweichung bedeutet, dass die Ergebnisse weniger zuverlässig sind
    und stärker von der spezifischen Datenaufteilung abhängen.
    
    Args:
        train_dataset: Vollständiger Trainingsdatensatz
        model_class: Die Modell-Klasse (z.B. BaselineCNN), die instanziiert werden soll.
                     Wird als Parameter übergeben, nicht hardcodiert.
        num_folds: Anzahl CV-Folds (Standard: 5)
        num_epochs: Epochen pro Fold (Standard: 60). Alle Folds trainieren exakt diese Anzahl Epochen.
        learning_rate: Lernrate
        batch_size: Batch-Größe
        img_size: Bildgröße für Modell-Initialisierung
        num_classes: Anzahl Klassen für Modell-Initialisierung
    
    Returns:
        dict: CV-Ergebnisse mit Mittelwerten und Standardabweichungen für alle Metriken.
              Struktur: {'metric_name': {'mean': float, 'std': float, 'values': list}}
              Metriken: train_accuracies, val_accuracies, train_losses, val_losses, train_f1s, val_f1s
    """
    
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'train_accuracies': [],
        'val_accuracies': [],
        'train_losses': [],
        'val_losses': [],
        'train_f1s': [],
        'val_f1s': []
    }
    
    print(f"Cross-Validation mit {num_folds} Folds")
    print(f"Jeder Fold trainiert exakt {num_epochs} Epochen (Early Stopping deaktiviert für Konsistenz)")
    print("=" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)))):
        print(f"\nFold {fold + 1}/{num_folds}")
        print("-" * 30)
        
        # Datensätze erstellen
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Modell erstellen - Modell-Klasse wird übergeben
        model = model_class(img_size=img_size, num_classes=num_classes).to(device)
        
        # Training OHNE Early Stopping für konsistente Epochenanzahl
        train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
            model, device, train_loader, val_loader, 
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping=False  # WICHTIG: Deaktiviert für konsistente Epochenanzahl
        )
        
        # Finale Metriken speichern
        cv_results['train_accuracies'].append(train_accs[-1])
        cv_results['val_accuracies'].append(val_accs[-1])
        cv_results['train_losses'].append(train_losses[-1])
        cv_results['val_losses'].append(val_losses[-1])
        cv_results['train_f1s'].append(train_f1s[-1])
        cv_results['val_f1s'].append(val_f1s[-1])
    
    # Statistiken berechnen
    cv_stats = {}
    for metric in cv_results:
        values = cv_results[metric]
        cv_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    print(f"\nCross-Validation Ergebnisse (Mittelwert ± Standardabweichung):")
    print("=" * 50)
    for metric, stats in cv_stats.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nHinweis: Die Standardabweichung zeigt die Variabilität zwischen den Folds.")
    print("Eine höhere Standardabweichung deutet auf größere Unsicherheit in den Ergebnissen hin.")
    
    return cv_stats