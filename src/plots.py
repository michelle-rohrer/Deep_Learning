import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#####################
# Klassenverteilung #
#####################

def plot_class_distribution(train_dataset, val_dataset, test_dataset):
    """
    Plottet die Klassenverteilung für Train, Validation und Test Sets
    
    Args:
        train_dataset: PyTorch Dataset (kann auch random_split Subset sein)
        val_dataset: PyTorch Dataset für Validierung
        test_dataset: PyTorch Dataset für Test
    """
    # Hilfsfunktion um Labels aus Dataset/Subset zu extrahieren
    def get_labels_and_classes(dataset):
        # Falls es ein Subset ist (von random_split),hole das ursprüngliche Dataset
        if hasattr(dataset, 'dataset'):
            original_dataset = dataset.dataset
            indices = dataset.indices
            # Labels für die ausgewählten Indices
            labels = [original_dataset.targets[i] for i in indices]
            class_names = list(original_dataset.class_to_idx.keys())
        else:
            # Direktes Dataset
            labels = dataset.targets
            class_names = list(dataset.class_to_idx.keys())
        
        return labels, class_names
    
    # Labels extrahieren
    train_labels, class_names = get_labels_and_classes(train_dataset)
    val_labels, _ = get_labels_and_classes(val_dataset)
    test_labels, _ = get_labels_and_classes(test_dataset)
    
    # Zählen
    train_df = pd.Series(train_labels).value_counts().sort_index()
    val_df = pd.Series(val_labels).value_counts().sort_index()
    test_df = pd.Series(test_labels).value_counts().sort_index()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Training-Set
    axes[0].bar(class_names, train_df.values, color="steelblue")
    axes[0].set_title("Training-Set")
    axes[0].set_xlabel("Klassen")
    axes[0].set_ylabel("Anzahl Bilder")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Validation-Set
    axes[1].bar(class_names, val_df.values, color="mediumseagreen")
    axes[1].set_title("Validation-Set")
    axes[1].set_xlabel("Klassen")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Test-Set
    axes[2].bar(class_names, test_df.values, color="darkorange")
    axes[2].set_title("Test-Set")
    axes[2].set_xlabel("Klassen")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

##################
# Image-Anzeigen #
##################

def plot_sample_images(dataset, class_to_idx=None, figsize=(12, 8), rows=2, cols=4, cmap="gray"):
    """
    Zeigt ein Beispielbild für jede Klasse aus einem PyTorch Dataset an.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        Das Dataset (z.B. ImageFolder oder Subset davon)
    class_to_idx : dict, optional
        Mapping von Klassennamen zu Indices. Wenn None, wird versucht es aus dataset.dataset zu holen
    figsize : tuple, optional
        Größe der Figure (Breite, Höhe), default (12, 8)
    rows : int, optional
        Anzahl der Zeilen im Subplot-Grid, default 2
    cols : int, optional
        Anzahl der Spalten im Subplot-Grid, default 4
    cmap : str, optional
        Colormap für die Darstellung, default "gray"
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Die erstellte Figure
    """
    # Versuche class_to_idx zu finden
    if class_to_idx is None:
        if hasattr(dataset, 'class_to_idx'):
            class_to_idx = dataset.class_to_idx
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'class_to_idx'):
            class_to_idx = dataset.dataset.class_to_idx
        else:
            raise ValueError("class_to_idx konnte nicht gefunden werden")
    
    # Invertiere das Mapping für Klassennamen
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # Sammle ein Beispielbild pro Klasse
    class_samples = {}
    
    for img, label in dataset:
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == num_classes:
            break
    
    # Plotten
    fig = plt.figure(figsize=figsize)
    
    for i, (class_idx, img_tensor) in enumerate(sorted(class_samples.items())):
        class_name = idx_to_class[class_idx]
        
        # Tensor zu numpy array konvertieren
        img = img_tensor.permute(1, 2, 0).numpy()
        
        # Falls Graustufenbild (1 Kanal)
        if img.shape[2] == 1:
            img = img.squeeze()
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(class_name)
        plt.axis("off")
    
    plt.tight_layout()
    return fig

######################
# Lernkurven plotten #
######################

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="Training Curves"):
    """
    Plottet die Lernkurven für Loss und Accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss Plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()