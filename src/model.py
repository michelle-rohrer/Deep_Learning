import torch.nn as nn
import torch.nn.functional as F

######################
# BaselineCNN Model #
#####################

class BaselineCNN(nn.Module):
    """
    Einfaches CNN als Baseline-Modell.
    
    Architektur:
    - 3 Convolutional Blocks (Conv2D + MaxPooling)
    - 2 Fully Connected Layers
    """
    
    def __init__(self, img_size=48, num_classes=7):
        super(BaselineCNN, self).__init__()
        
        # === FEATURE EXTRAKTION ===
        # Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # === KLASSIFIKATION ===
        # Berechne die Größe nach allen Pooling-Layers
        # img_size / 2 / 2 / 2 = img_size / 8
        flatten_size = 64 * (img_size // 8) * (img_size // 8)
        
        self.fc1 = nn.Linear(flatten_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


######################
# FlexibleCNN Model #
######################

class FlexibleCNN(nn.Module):
    """
    Flexibles CNN-Modell für Hyperparameter-Experimente.
    
    Ermöglicht Variation von:
    - Anzahl Conv-Layer (Modelltiefe)
    - Anzahl Filter pro Layer (Modellbreite)
    - Anzahl Neuronen in FC-Layer (FC-Breite)
    - Kernel Size
    - Pooling Type
    - Selektives Pooling (welche Layer gepoolt werden)
    """
    def __init__(self, img_size=48, num_classes=7, 
                 num_conv_layers=3, filters_per_layer=[16, 32, 64],
                 kernel_size=3, fc_units=64, pooling_type='max',
                 pool_after_layers=None):
        """
        Args:
            pool_after_layers: Liste von Layer-Indizes (0-basiert), nach denen gepoolt werden soll.
                              Wenn None, wird nach jedem Layer gepoolt (Standard).
                              Beispiel: [0, 1, 2] bedeutet Pooling nach Layer 1, 2, 3.
        """
        super(FlexibleCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.img_size = img_size
        
        # Standard: Pooling nach jedem Layer, wenn nicht spezifiziert
        if pool_after_layers is None:
            pool_after_layers = list(range(num_conv_layers))
        self.pool_after_layers = set(pool_after_layers)
        
        # Convolutional Layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = 1
        num_poolings = 0
        for i in range(num_conv_layers):
            # Filter-Anzahl: Verwende filters_per_layer[i] oder letztes Element wenn zu wenig angegeben
            out_channels = filters_per_layer[i] if i < len(filters_per_layer) else filters_per_layer[-1]
            
            # Conv Layer mit Padding um Größe zu erhalten
            padding = kernel_size // 2
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            )
            
            # Pooling nur wenn spezifiziert
            if i in self.pool_after_layers:
                if pooling_type == 'max':
                    self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pooling_type == 'avg':
                    self.pool_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    raise ValueError(f"Unbekannter pooling_type: {pooling_type}")
                num_poolings += 1
            else:
                # Placeholder für Layer ohne Pooling
                self.pool_layers.append(None)
            
            in_channels = out_channels
        
        # Flatten size berechnen
        # Nach jedem Pooling: img_size / 2
        # Nach num_poolings Poolings: img_size / (2^num_poolings)
        final_size = img_size // (2 ** num_poolings)
        flatten_size = in_channels * final_size * final_size
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flatten_size, fc_units)
        self.fc2 = nn.Linear(fc_units, num_classes)
    
    def forward(self, x):
        # Convolutional Blocks
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = F.relu(x)
            # Pooling nur wenn vorhanden
            if self.pool_layers[i] is not None:
                x = self.pool_layers[i](x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


#############################
# Convenience-Funktionen    #
#############################

def create_deeper_model(img_size=48, num_classes=7, num_layers=4):
    """
    Erstellt ein tieferes Modell mit mehr Conv-Layern.
    
    Args:
        num_layers: Anzahl Conv-Layer (2, 3, 4, 5, 6)
    """
    # Filter-Strategie: Erweitere die Filter-Anzahl mit mehr Layern
    filter_configs = {
        2: [32, 64],
        3: [16, 32, 64],  # Baseline
        4: [16, 32, 64, 128],
        5: [16, 32, 64, 128, 256],
        6: [16, 32, 64, 128, 256, 512]
    }
    
    filters = filter_configs.get(num_layers, [16, 32, 64, 128, 256, 512][:num_layers])
    
    return FlexibleCNN(
        img_size=img_size, 
        num_classes=num_classes,
        num_conv_layers=num_layers, 
        filters_per_layer=filters
    )


def create_wider_model(img_size=48, num_classes=7, filter_multiplier=1):
    """
    Erstellt ein breiteres Modell mit mehr Filtern pro Layer.
    
    Args:
        filter_multiplier: Multiplikator für Filter-Anzahl (1=Baseline, 2=doppelt, etc.)
    """
    base_filters = [16, 32, 64]
    filters = [f * filter_multiplier for f in base_filters]
    
    return FlexibleCNN(
        img_size=img_size, 
        num_classes=num_classes,
        filters_per_layer=filters
    )


def create_model_with_fc_width(img_size=48, num_classes=7, fc_units=64):
    """
    Erstellt ein Modell mit variabler FC-Layer Breite.
    
    Args:
        fc_units: Anzahl Neuronen im ersten FC-Layer
    """
    return FlexibleCNN(
        img_size=img_size, 
        num_classes=num_classes,
        fc_units=fc_units
    )