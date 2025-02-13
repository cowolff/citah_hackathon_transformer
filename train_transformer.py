import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import VideoDataLoader
from model import DualProjectionTransformer
from vae_model import VAE
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import json

def count_parameters(model):
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_parameters = sum(p.numel() for p in model.parameters())
    return all_parameters, trainable_parameters

def train_model(model, train_loader, val_loader, device, class_weights=[1, 1], epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)

    all_params, trainable_params = count_parameters(model)
    print(f"The model has {all_params:,} parameters in total, with {trainable_params:,} trainable parameters.")
    
    history = {
        'train_loss': [],
        'train_acc_class_0': [],
        'train_acc_class_1': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'train_acc': []
    }
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = [0] * 2  # Assuming 2 classes
        total = [0] * 2
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames, floorplans, labels in progress_bar:
            frames, floorplans, labels = frames.to(device, dtype=torch.float32), \
                                        floorplans.to(device, dtype=torch.float32), \
                                        labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(floorplans, frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                correct[label] += (predicted[i] == label).item()
            
            train_acc = [100. * correct[i] / total[i] if total[i] > 0 else 0 for i in range(2)]
            total_acc = sum(correct) / sum(total)
            progress_bar.set_postfix(loss=running_loss / (sum(total) // labels.size(0)), acc=total_acc,
                                     acc_class_0=train_acc[0], acc_class_1=train_acc[1])
            
            history['train_acc'].append(total_acc)
            history['train_loss'].append(loss.item())
        
        train_loss = running_loss / len(train_loader)
        train_acc = [100. * correct[i] / total[i] if total[i] > 0 else 0 for i in range(2)]
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Calculate precision and recall
        all_labels = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for frames, floorplans, labels in val_loader:
                frames, floorplans, labels = frames.to(device, dtype=torch.float32), \
                                            floorplans.to(device, dtype=torch.float32), \
                                            labels.to(device, dtype=torch.long)
                outputs = model(floorplans, frames)
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        val_precision = precision_score(all_labels, all_predictions, average='weighted')
        val_recall = recall_score(all_labels, all_predictions, average='weighted')
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc Class 0: {train_acc[0]:.2f}%, Train Acc Class 1: {train_acc[1]:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Precision: {val_precision:.2f}, Val Recall: {val_recall:.2f}")
        
        history['train_acc_class_0'].append(train_acc[0])
        history['train_acc_class_1'].append(train_acc[1])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
    
    torch.save(model.state_dict(), "dual_projection_transformer.pth")
    print("Model saved successfully.")
    
    return history


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, floorplans, labels in val_loader:
            frames, floorplans, labels = frames.to(device, dtype=torch.float32), \
                                        floorplans.to(device, dtype=torch.float32), \
                                        labels.to(device, dtype=torch.long)
            
            outputs = model(floorplans, frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def plot_loss_accuracy(history):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], label='Train Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(history['train_acc'], label='Train Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Loss and Accuracy')
    plt.savefig("figures/loss_accuracy.pdf")
    plt.show()

# Depending on whether you are on mac or linux, you may need to change the device to "cuda" or "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Load data
data_filepath = "pikk_hackathon/frames_data.h5"
dataset = VideoDataLoader(data_filepath)

label_share = dataset.label_share
class_weights = [1 / label_share[label] for label in range(2)]

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

image_vae = VAE()
floorplan_vae = VAE()

# Load model
image_vae.load_state_dict(torch.load("models/vae_full.pth"))
floorplan_vae.load_state_dict(torch.load("models/vae_floorline.pth"))

model = DualProjectionTransformer(image_vae, floorplan_vae, embed_dim=32, num_heads=8, num_layers=3, num_classes=2)

history = train_model(model, train_loader, val_loader, device, class_weights=class_weights, epochs=1, lr=0.0001)

plot_loss_accuracy(history)

with open("results/history.json", "w") as f:
    json.dump(history, f)