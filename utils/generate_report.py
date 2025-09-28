import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from utils.graph_generator import generate_confusion_matrix

def save_training_results(train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_acc, y_true, y_pred, num_epochs, lr, folder_name):
    # ---- 1. Save plots ----
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = f"{folder_name}/loss_plot.png"
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    acc_plot_path = f"{folder_name}/accuracy_plot.png"
    plt.savefig(acc_plot_path)
    plt.close()

    # ---- Generate and save confusion matrix ----
    cm_plot_path = f"{folder_name}/confusion_matrix.png"
    generate_confusion_matrix(y_true, y_pred, folder_name=folder_name, class_names=['NORMAL', 'PNEUMONIA'])
    plt.close()

    # ---- 2. Create PDF ----
    pdf_filename = f"{folder_name}/training_report_epoch{num_epochs}_lr{lr}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 40, f"Training Report - {num_epochs} Epochs, LR={lr}")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 80, f"Test Loss: {test_loss:.4f}")
    c.drawString(40, height - 100, f"Test Accuracy: {test_acc:.2f}%")

    c.drawImage(loss_plot_path, 50, height - 350, width=400, height=200)
    c.drawImage(acc_plot_path, 50, height - 570, width=400, height=200)
    c.drawImage(f"{folder_name}/confusion_matrix.png", 50, height - 800, width=400, height=200)

    c.save()
    print(f"✅ Training report saved to {pdf_filename}")
