import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mymodels.CNN import CNN, train_cnn_model, test_model
from utils.generate_report import save_training_results
import os
from typing import List
import datetime

def main(epochs:List[int], lrs: List[float],batch_size:int=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)                 # should print cuda
    print(torch.cuda.is_available())  # should be True

    transform = transforms.Compose([
    transforms.Resize((1500, 1500)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    train_dataset = datasets.ImageFolder(root="chest_xray/train", transform=transform)
    val_dataset   = datasets.ImageFolder(root="chest_xray/val", transform=transform)
    test_dataset  = datasets.ImageFolder(root="chest_xray/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    base_folder = "outputs"
    os.makedirs(base_folder, exist_ok=True)

    for epoch in epochs:
        for lr in lrs:
            run_folder = f"{base_folder}/epoch{epoch}_lr{lr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(run_folder, exist_ok=True)

            best_model_state, train_losses, val_losses, train_accuracies, val_accuracies = train_cnn_model(
                train_loader,
                val_loader,
                device=device,
                num_epochs=epoch,
                lr=lr
            )

            model = CNN(num_classes=2)
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, f"{run_folder}/best_model.pth")

            test_loss, test_acc, y_true, y_pred = test_model(model, test_loader, device=device)

            save_training_results(
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies, 
                test_loss,
                test_acc,
                y_true,
                y_pred,
                epoch,
                lr,
                run_folder
            )

if __name__ == "__main__":
    main([10, 20], [0.002, 0.005], batch_size=4)
