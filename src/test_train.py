import torch
import torch.nn as nn

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