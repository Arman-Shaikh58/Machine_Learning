import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------
# 1. Transformations
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((2000, 2000)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ------------------------------
# 2. Load Dataset
# ------------------------------
train_dataset = datasets.ImageFolder(root="chest_xray/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="chest_xray/val", transform=transform)
test_dataset  = datasets.ImageFolder(root="chest_xray/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ------------------------------
# 3. CNN Model
# ------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 1000->500

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 500->250

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)  # 250->125
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16,16))
        self.fc_layers = nn.Sequential(
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layers(x)
        return x

# ------------------------------
# 4. Training Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

num_epochs = 10

# Lists to store loss and accuracy
train_losses = []
val_accuracies = []
logss=[]
# ------------------------------
# 5. Training Loop
# ------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss/len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100*correct/total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
    logss.append(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
# ------------------------------
# 6. Test Accuracy
# ------------------------------
model.eval()
correct, total = 0,0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = 100*correct/total
print(f"Test Accuracy: {test_acc:.2f}%")


# ------------------------------
# 7. Save model
# ------------------------------
torch.save(model.state_dict(), "phenomena_vs_normal_cnn.pth")

# ------------------------------
# 8. Plot Loss and Accuracy
# ------------------------------
plt.figure(figsize=(10,5))
plt.plot(range(1,num_epochs+1), train_losses, label="Training Loss")
plt.plot(range(1,num_epochs+1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy (%)")
plt.title("Training Loss and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("loss_accuracy_plot.png")  # save plot as image
plt.show()
