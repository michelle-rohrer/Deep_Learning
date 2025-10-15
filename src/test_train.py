import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import wandb

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

def train_model(model, device, train_loader, val_loader, num_epochs=50, learning_rate=0.01, batch_size=64, use_wandb=False, run_name=None, early_stopping=True, patience=10, min_delta=0.001):
    """
    Training des Basismodells mit SGD (ohne Momentum), ohne Regularisierung, ohne Batchnorm.
    
    Args:
        model: Das zu trainierende Modell
        device: CUDA oder CPU
        train_loader: DataLoader für Training
        val_loader: DataLoader für Validation
        num_epochs: Anzahl Epochen
        learning_rate: Lernrate für SGD
        batch_size: Batch-Größe
        use_wandb: Ob wandb logging aktiviert werden soll
        run_name: Name für den wandb run
        early_stopping: Ob Early Stopping aktiviert werden soll
        patience: Anzahl Epochen ohne Verbesserung vor Stopp
        min_delta: Minimale Verbesserung um als besser zu gelten
    
    Returns:
        train_losses, val_losses, train_accs, val_accs: Listen mit Metriken pro Epoche
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
    
    # Listen für Metriken
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Metriken berechnen
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Speichern
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
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
        
        # wandb logging
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
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
    
    return train_losses, val_losses, train_accs, val_accs

#########################
# Hyperparameter-Tuning #
#########################

def hyperparameter_tuning_with_wandb(train_dataset, val_dataset, learning_rates, batch_sizes, num_epochs=30):
    """
    Hyperparameter-Tuning mit wandb Integration für besseres Experiment-Tracking.
    
    Args:
        train_dataset: Trainingsdatensatz
        val_dataset: Validierungsdatensatz
        learning_rates: Liste der zu testenden Lernraten
        batch_sizes: Liste der zu testenden Batch-Größen
        num_epochs: Anzahl Epochen pro Konfiguration
    
    Returns:
        dict: Ergebnisse aller Konfigurationen
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
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
            
            # Neues Modell für jeden Test
            model = BaselineCNN(img_size=48, num_classes=7).to(device)
            
            # Training mit wandb und Early Stopping
            train_losses, val_losses, train_accs, val_accs = train_model(
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
            
            # Ergebnisse speichern
            results[run_name] = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'final_train_acc': train_accs[-1],
                'final_val_acc': val_accs[-1],
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1]
            }
            
            print(f"Finale Validation Accuracy: {val_accs[-1]:.2f}%")
            print(f"Finale Validation Loss: {val_losses[-1]:.4f}")
    
    return results