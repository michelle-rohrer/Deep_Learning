import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import wandb
import os

from src.model import BaselineCNN

####################
# Overfitting Test #
####################

def overfitting_test_batch(model, device, train_loader, num_epochs=500):
    """
    Testet ob das Modell einen Batch perfekt lernen kann.
    """

    # Einen Batch holen
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accuracy berechnen
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)
        
        # Alle 20 Epochen ausgeben
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f} | "
                  f"Accuracy: {accuracy:.4f} ({int(accuracy*len(labels))}/{len(labels)} correct)")
    
    # Finaler Test
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        final_accuracy = (predicted == labels).sum().item() / len(labels)
        final_loss = criterion(outputs, labels).item()
    
    return final_loss, final_accuracy


########################
# Training des Modells #
########################

def train_model(model, device, train_loader, val_loader, num_epochs=50, learning_rate=0.01, batch_size=64, use_wandb=False, run_name=None, early_stopping=True, patience=10, min_delta=0.001, save_path=None):
    """
    Training des Basismodells mit SGD (ohne Momentum), ohne Regularisierung, ohne Batchnorm.
    
    Diese Funktion führt automatisch einen Overfitting-Test durch, wenn der train_loader nur einen Batch enthält.
    Das Modell wird automatisch nach Epoche 60 gespeichert (falls num_epochs >= 60).
    
    **Overfitting-Test:**
    Wenn train_loader nur 1 Batch enthält (len(train_loader) == 1), wird automatisch ein Overfitting-Test durchgeführt.
    Dies testet, ob das Modell in der Lage ist, einen einzelnen Batch perfekt zu lernen.
    Ein erfolgreicher Overfitting-Test zeigt, dass das Modell lernen kann (nicht auswendig lernt).
    
    **Automatisches Speichern:**
    Wenn save_path angegeben ist und num_epochs >= 60, wird das Modell automatisch nach Epoche 60 gespeichert.
    Dies dient als Referenzpunkt für konsistente Vergleiche.
    
    **Metriken:**
    Die Funktion berechnet und gibt zurück:
    - Loss (Training und Validation)
    - Accuracy (Training und Validation)
    - F1-Score (Training und Validation, weighted average)
    
    Args:
        model: Das zu trainierende Modell (wird direkt übergeben, nicht aus Code erstellt)
        device: CUDA oder CPU
        train_loader: DataLoader für Training. Wenn nur 1 Batch, wird automatisch Overfitting-Test durchgeführt.
        val_loader: DataLoader für Validation
        num_epochs: Anzahl Epochen (Standard: 50, empfohlen: 60 für Referenz)
        learning_rate: Lernrate für SGD
        batch_size: Batch-Größe
        use_wandb: Ob wandb logging aktiviert werden soll
        run_name: Name für den wandb run
        early_stopping: Ob Early Stopping aktiviert werden soll
        patience: Anzahl Epochen ohne Verbesserung vor Stopp
        min_delta: Minimale Verbesserung um als besser zu gelten
        save_path: Pfad zum Speichern des Modells nach Epoche 60 (optional)
    
    Returns:
        tuple: (train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
            Listen mit Metriken pro Epoche für alle 6 Metriken
    """
    
    # SGD Optimizer ohne Momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    criterion = nn.CrossEntropyLoss()
    
    # wandb initialisieren falls gewünscht
    if use_wandb:
        wandb.init(
            project="emotion-recognition-baseline",
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "optimizer": "SGD",
                "momentum": 0,
                "architecture": "BaselineCNN",
                "dataset": "FER-2013"
            }
        )
    
    # Overfitting-Test erkennen: Wenn train_loader nur 1 Batch hat
    is_overfitting_test = len(train_loader) == 1
    if is_overfitting_test:
        print("=" * 60)
        print("OVERFITTING-TEST MODUS: Train_loader enthält nur 1 Batch")
        print("Das Modell wird auf einem einzelnen Batch trainiert, um zu testen,")
        print("ob es in der Lage ist, die Daten perfekt zu lernen (Overfitting-Test).")
        print("=" * 60)
    
    # Listen für Metriken
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    
    # Early Stopping Variablen
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"Training startet: {num_epochs} Epochen, LR={learning_rate}, Batch={batch_size}")
    if early_stopping:
        print(f"Early Stopping: patience={patience}, min_delta={min_delta}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metriken sammeln
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Für F1-Score sammeln
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Für F1-Score sammeln
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        # Metriken berechnen
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # F1-Score berechnen
        train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')
        
        # Speichern
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        # Early Stopping Check
        if early_stopping:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                # Bestes Modell speichern (optional)
                # torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
        
        # Automatisches Speichern nach Epoche 60
        if (epoch + 1) == 60 and save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"\nModell nach Epoche 60 gespeichert: {save_path}")
        
        # wandb logging
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter
            })
        
        # Ausgabe
        if (epoch + 1) % 5 == 0 or epoch == 0:
            early_stop_info = f" | Patience: {patience_counter}/{patience}" if early_stopping else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%{early_stop_info}")
        
        # Early Stopping Check
        if early_stopping and patience_counter >= patience:
            print(f"\nEarly Stopping nach {epoch + 1} Epochen!")
            print(f"Beste Validation Loss: {best_val_loss:.4f} in Epoche {best_epoch}")
            break
    
    # wandb run beenden
    if use_wandb:
        wandb.finish()
    
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s

#########################
# Hyperparameter-Tuning #
#########################

def hyperparameter_tuning_with_wandb(train_dataset, val_dataset, learning_rates, batch_sizes, model_class, num_epochs=60, img_size=48, num_classes=7):
    """
    Hyperparameter-Tuning mit wandb Integration für besseres Experiment-Tracking.
    
    **Was wird gemacht:**
    - Testet alle Kombinationen von Lernraten und Batch-Größen
    - Erstellt für jede Konfiguration ein neues Modell aus der übergebenen Modell-Klasse
    - Trainiert jedes Modell mit Early Stopping
    - Speichert Trainingskurven (Loss, Accuracy, F1-Score) für Training und Validation
    - Findet die beste Performance pro Konfiguration
    
    **Ergebnisse:**
    Das zurückgegebene Dictionary enthält für jede Konfiguration:
    - Trainingskurven (train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    - Finale Metriken (nach allen Epochen)
    - Beste Metriken (beste Validation Accuracy während Training)
    - Anzahl tatsächlich trainierter Epochen
    
    **Hinweis zu kleinen Datasets:**
    Bei kleinen Datasets kann das Training schneller konvergieren. Early Stopping passt sich automatisch an.
    
    Args:
        train_dataset: Trainingsdatensatz
        val_dataset: Validierungsdatensatz
        learning_rates: Liste der zu testenden Lernraten
        batch_sizes: Liste der zu testenden Batch-Größen
        model_class: Die Modell-Klasse (z.B. BaselineCNN), die instanziiert werden soll.
                     Wird als Parameter übergeben, nicht hardcodiert.
        num_epochs: Anzahl Epochen pro Konfiguration (Standard: 60 für Referenz)
        img_size: Bildgröße für Modell-Initialisierung
        num_classes: Anzahl Klassen für Modell-Initialisierung
    
    Returns:
        dict: Ergebnisse aller Konfigurationen mit Trainingskurven und Metriken.
              Key-Format: "LR_{lr}_Batch_{batch_size}"
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    print(f"=== Hyperparameter-Tuning mit wandb ===")
    print(f"Teste {len(learning_rates)} Lernraten × {len(batch_sizes)} Batch-Größen = {len(learning_rates) * len(batch_sizes)} Konfigurationen")
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            run_name = f"LR_{lr}_Batch_{batch_size}"
            print(f"\n--- {run_name} ---")
            
            # Neue DataLoader mit aktueller Batch-Größe
            # Optimale num_workers für Multi-Core (M5): Anzahl Cores - 1
            import multiprocessing
            optimal_workers = max(1, multiprocessing.cpu_count() - 1)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=optimal_workers,
                pin_memory=True  # Beschleunigt Transfer zu GPU/MPS
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=optimal_workers,
                pin_memory=True
            )
            
            # Neues Modell für jeden Test - Modell-Klasse wird übergeben
            model = model_class(img_size=img_size, num_classes=num_classes).to(device)
            
            # Training mit wandb und Early Stopping
            train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
                model=model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=lr,
                batch_size=batch_size,
                use_wandb=True,
                run_name=run_name,
                early_stopping=True,
                patience=8,  # Etwas weniger patience für Hyperparameter-Tuning
                min_delta=0.001
            )
            
            # Beste Performance finden (höchste Validation Accuracy)
            best_epoch_idx = np.argmax(val_accs)
            best_val_acc = val_accs[best_epoch_idx]
            best_val_f1 = val_f1s[best_epoch_idx]
            
            # Ergebnisse speichern
            results[run_name] = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'num_epochs': len(train_accs),  # Tatsächliche Anzahl Epochen (kann durch Early Stopping reduziert sein)
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
                'final_train_acc': train_accs[-1],
                'final_val_acc': val_accs[-1],
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_train_f1': train_f1s[-1],
                'final_val_f1': val_f1s[-1],
                'best_epoch': best_epoch_idx + 1,
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'best_train_acc': train_accs[best_epoch_idx],
                'best_train_f1': train_f1s[best_epoch_idx]
            }
            
            print(f"Finale Validation Accuracy: {val_accs[-1]:.2f}% | F1-Score: {val_f1s[-1]:.4f}")
            print(f"Beste Validation Accuracy: {best_val_acc:.2f}% (Epoche {best_epoch_idx + 1}) | F1-Score: {best_val_f1:.4f}")
            print(f"Finale Validation Loss: {val_losses[-1]:.4f}")
    
    return results