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