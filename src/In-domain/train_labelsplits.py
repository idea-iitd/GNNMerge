# This file is used to train the label split models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from models import GNNMLP, GNNComplete, SageBackbone, GCNBackbone
from torch.optim import Adam
import sys
import time
import os
import json
from datetime import datetime
import argparse

def setup_args():
    parser = argparse.ArgumentParser(description='Train label split models')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['gnn', 'sage'], help='Model architecture to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model1_save_path', type=str, required=True, help='Path to save Model 1')
    parser.add_argument('--model2_save_path', type=str, required=True, help='Path to save Model 2')
    parser.add_argument('--logs_path', type=str, required=True, help='Directory to save training logs')
    return parser.parse_args()

def setup_dataset(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.load(data_path, map_location="cpu")
    N = dataset.num_nodes
    labels = len(dataset.label_names)
    input_dim = dataset.x.size(1)
    
    print(f"Dataset info:")
    print(f"Number of nodes: {N}")
    print(f"Number of labels: {labels}")
    print(f"Input feature dimension: {input_dim}")
    
    return dataset, device, N, labels, input_dim

def setup_models(model_name, input_dim, labels, device):
    hidden_dim = 128
    
    if model_name == 'gcn':
        model1_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        model2_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == "sage":
        model1_backbone = SageBackbone(input_dim, hidden_dim).to(device)
        model2_backbone = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    model1_mlp = GNNMLP(hidden_dim, (labels+1)//2).to(device)
    model2_mlp = GNNMLP(hidden_dim, labels//2).to(device)
    
    model1 = GNNComplete(model1_backbone, model1_mlp).to(device)
    model2 = GNNComplete(model2_backbone, model2_mlp).to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    return model1, model2, optimizer1, optimizer2, criterion

def setup_masks(dataset, N, labels):
    classes_set1 = set(range(0, (labels+1)//2))
    classes_set2 = set(range((labels+1)//2, labels))
    
    train_mask1 = torch.zeros(N, dtype=torch.bool, device='cpu')
    train_mask2 = torch.zeros(N, dtype=torch.bool, device='cpu')
    test_mask1 = torch.zeros(N, dtype=torch.bool, device='cpu')
    test_mask2 = torch.zeros(N, dtype=torch.bool, device='cpu')
    val_mask1 = torch.zeros(N, dtype=torch.bool, device='cpu')
    val_mask2 = torch.zeros(N, dtype=torch.bool, device='cpu')
    
    train_indices = dataset.train_masks[0]
    val_indices = dataset.val_masks[0]
    test_indices = dataset.test_masks[0]
    
    original_y = dataset.y.clone()
    
    # Setup masks based on labels
    for idx in train_indices:
        label = dataset.y[idx].item()
        if label in classes_set1:
            train_mask1[idx] = True
        elif label in classes_set2:
            train_mask2[idx] = True
            
    for idx in test_indices:
        label = dataset.y[idx].item()
        if label in classes_set1:
            test_mask1[idx] = True
        elif label in classes_set2:
            test_mask2[idx] = True

    for idx in val_indices:
        label = dataset.y[idx].item()
        if label in classes_set1:
            val_mask1[idx] = True
        elif label in classes_set2:
            val_mask2[idx] = True
    
    # Create label mapping for second model
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(list(classes_set2)))}
    dataset.y = original_y.clone()
    for idx in range(len(dataset.y)):
        if (train_mask2[idx] or test_mask2[idx] or val_mask2[idx]):
            dataset.y[idx] = label_mapping[dataset.y[idx].item()]
            
    return (train_mask1, train_mask2, val_mask1, val_mask2, 
            test_mask1, test_mask2, classes_set1, classes_set2)

def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), out

def evaluate(model, data, mask, criterion):
    model.eval()
    out = model(data)
    loss = criterion(out[mask], data.y[mask]).item()
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    samples = mask.sum()
    acc = float(correct)/float(samples)
    return acc, loss

def train_model(model_id, model, dataset, train_mask, val_mask, test_mask, optimizer, criterion, dataset_name, model_name, args):
    num_epochs = 500
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize logs
    logs = {
        'model_name': f"{dataset_name}_{model_name}_{model_id}",
        'epochs': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'time_per_epoch': [],
        'best_epoch': 0,
        'best_val_acc': 0,
        'best_test_acc': 0,
        'total_training_time': 0
    }
    
    best_val_acc = 0
    best_test_acc = 0
    total_time = 0
    
    print(f"\nStarting training for Model {model_id}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, _ = train(model, dataset, train_mask, optimizer, criterion)
        
        # Evaluate
        train_acc, train_loss = evaluate(model, dataset, train_mask, criterion)
        val_acc, val_loss = evaluate(model, dataset, val_mask, criterion)
        test_acc, test_loss = evaluate(model, dataset, test_mask, criterion)
        
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        # Update logs
        logs['epochs'].append(epoch)
        logs['train_acc'].append(train_acc)
        logs['val_acc'].append(val_acc)
        logs['test_acc'].append(test_acc)
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['test_loss'].append(test_loss)
        logs['time_per_epoch'].append(epoch_time)
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            logs['best_epoch'] = epoch
            logs['best_val_acc'] = val_acc
            logs['best_test_acc'] = test_acc
            # Update model saving path based on command line argument
            save_path = args.model1_save_path if model_id == 1 else args.model2_save_path
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc
            }, save_path)
            print(f"Saved best model checkpoint to {save_path}")
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs-1}:")
        print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Print best accuracies every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"\nBest so far (Model {model_id}):")
            print(f"Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f} (Epoch {logs['best_epoch']})")
            print(f"Average epoch time: {total_time/(epoch+1):.2f}s")
            print("-" * 50)
    
    logs['total_training_time'] = total_time
    
    # Update logs saving path based on command line argument
    log_file = os.path.join(args.logs_path, f'{dataset_name}_{model_name}_{model_id}_{timestamp}.json')
    os.makedirs(args.logs_path, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Saved training logs to {log_file}")
    
    return logs

def training(dataset, model1, model2, masks, optimizer1, optimizer2, criterion, args):
    train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2, classes_set1, classes_set2 = masks
    
    # Train Model 1 (First half of classes)
    print("\nTraining Model 1 (First half of classes)")
    logs1 = train_model(
        model_id=1,
        model=model1,
        dataset=dataset,
        train_mask=train_mask1,
        val_mask=val_mask1,
        test_mask=test_mask1,
        optimizer=optimizer1,
        criterion=criterion,
        dataset_name=args.dataset,
        model_name=args.model,
        args=args
    )
    
    # Train Model 2 (Second half of classes)
    print("\nTraining Model 2 (Second half of classes)")
    logs2 = train_model(
        model_id=2,
        model=model2,
        dataset=dataset,
        train_mask=train_mask2,
        val_mask=val_mask2,
        test_mask=test_mask2,
        optimizer=optimizer2,
        criterion=criterion,
        dataset_name=args.dataset,
        model_name=args.model,
        args=args
    )
    
    # Print final summary
    print("\nTraining completed!")
    print("\nModel 1 Summary:")
    print(f"Best val acc: {logs1['best_val_acc']:.4f} (Epoch {logs1['best_epoch']})")
    print(f"Test acc at best val: {logs1['best_test_acc']:.4f}")
    print(f"Total training time: {logs1['total_training_time']:.2f}s")
    print(f"Average epoch time: {logs1['total_training_time']/500:.2f}s")
    print(f"Model saved at: {args.model1_save_path}")
    
    print("\nModel 2 Summary:")
    print(f"Best val acc: {logs2['best_val_acc']:.4f} (Epoch {logs2['best_epoch']})")
    print(f"Test acc at best val: {logs2['best_test_acc']:.4f}")
    print(f"Total training time: {logs2['total_training_time']:.2f}s")
    print(f"Average epoch time: {logs2['total_training_time']/500:.2f}s")
    print(f"Model saved at: {args.model2_save_path}")
    
    print(f"\nAll training logs saved in: {args.logs_path}")

def print_split_info(dataset, masks):
    train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2, classes_set1, classes_set2 = masks
    
    print("\nLabel split info:")
    print(f"Total classes: {len(classes_set1) + len(classes_set2)}")
    print(f"Model 1 classes: {sorted(list(classes_set1))}")
    print(f"Model 2 classes: {sorted(list(classes_set2))}")

    print("\nLabel distribution after splitting:")
    print("Model 1:")
    print(f"Train labels: {torch.unique(dataset.y[train_mask1]).tolist()}")
    print(f"Val labels: {torch.unique(dataset.y[val_mask1]).tolist()}")
    print(f"Test labels: {torch.unique(dataset.y[test_mask1]).tolist()}")
    print(f"Number of nodes - Train: {train_mask1.sum().item()}, Val: {val_mask1.sum().item()}, Test: {test_mask1.sum().item()}")

    print("\nModel 2:")
    print(f"Train labels: {torch.unique(dataset.y[train_mask2]).tolist()}")
    print(f"Val labels: {torch.unique(dataset.y[val_mask2]).tolist()}")
    print(f"Test labels: {torch.unique(dataset.y[test_mask2]).tolist()}")
    print(f"Number of nodes - Train: {train_mask2.sum().item()}, Val: {val_mask2.sum().item()}, Test: {test_mask2.sum().item()}")

    print("\nTotal nodes in splits:")
    print(f"Train: {train_mask1.sum().item() + train_mask2.sum().item()}")
    print(f"Val: {val_mask1.sum().item() + val_mask2.sum().item()}")
    print(f"Test: {test_mask1.sum().item() + test_mask2.sum().item()}")
    print(f"Total nodes in dataset: {dataset.num_nodes}")

def main():
    args = setup_args()
    print("\nConfiguration:")
    print(f"Dataset: {args.dataset}")
    print(f"Model architecture: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Model 1 save path: {args.model1_save_path}")
    print(f"Model 2 save path: {args.model2_save_path}")
    print(f"Logs path: {args.logs_path}\n")
    
    dataset, device, N, labels, input_dim = setup_dataset(args.data_path)
    model1, model2, optimizer1, optimizer2, criterion = setup_models(args.model, input_dim, labels, device)
    
    masks = setup_masks(dataset, N, labels)
    train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2, classes_set1, classes_set2 = masks
    
    # Move data to device
    dataset = dataset.to(device)
    train_mask1, train_mask2 = train_mask1.to(device), train_mask2.to(device)
    val_mask1, val_mask2 = val_mask1.to(device), val_mask2.to(device)
    test_mask1, test_mask2 = test_mask1.to(device), test_mask2.to(device)
    
    print_split_info(dataset, masks)
    training(dataset, model1, model2, masks, optimizer1, optimizer2, criterion, args)

if __name__ == "__main__":
    main()

