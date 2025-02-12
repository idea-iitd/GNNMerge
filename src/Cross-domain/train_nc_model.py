import torch
import torch.nn as nn
import os
import json
import time
from datetime import datetime
import argparse
from models import GCNBackbone, GNNMLP, GNNComplete, SageBackbone

def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN models for node classification')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=True,
                      choices=['gcn', 'sage'],
                      help='Model architecture to use')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset file')
    parser.add_argument('--logs_path', type=str, required=True,
                      help='Directory to save logs')
    parser.add_argument('--model_save_path', type=str, required=True,
                      help='Directory to save model checkpoints')
    return parser.parse_args()

def setup_model(model_name, input_dim, hidden_dim, labels, device):
    if model_name == 'gcn':
        backbone = GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == "sage":
        backbone = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    mlp = GNNMLP(hidden_dim, labels).to(device)
    return GNNComplete(backbone, mlp).to(device)

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
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[mask], data.y[mask]).item()
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        samples = mask.sum()
        acc = float(correct)/float(samples)
    return acc, loss

def training(args, model, dataset, optimizer, criterion):
    # Create required directories
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.logs_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = {
        'model_name': f"{args.dataset_name}_{args.model_name}",
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
    total_time = 0
    
    print("\nStarting training")
    print("-" * 50)
    
    for epoch in range(500):
        epoch_start_time = time.time()
        
        # Train
        train_loss, _ = train(model, dataset, dataset.train_masks[0], optimizer, criterion)
        
        # Evaluate
        train_acc, train_loss = evaluate(model, dataset, dataset.train_masks[0], criterion)
        val_acc, val_loss = evaluate(model, dataset, dataset.val_masks[0], criterion)
        test_acc, test_loss = evaluate(model, dataset, dataset.test_masks[0], criterion)
        
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logs['best_epoch'] = epoch
            logs['best_val_acc'] = val_acc
            logs['best_test_acc'] = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc
            }, os.path.join(args.model_save_path, f'{args.dataset_name}_{args.model_name}.pt'))
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/500:")
            print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
            print(f"Best val acc: {best_val_acc:.4f} (Epoch {logs['best_epoch']})")
            print(f"Average epoch time: {total_time/(epoch+1):.2f}s")
            print("-" * 50)
    
    logs['total_training_time'] = total_time
    
    # Save logs
    log_file = os.path.join(args.logs_path, f'{args.dataset_name}_{args.model_name}_{timestamp}.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Best val acc: {logs['best_val_acc']:.4f} (Epoch {logs['best_epoch']})")
    print(f"Test acc at best val: {logs['best_test_acc']:.4f}")
    
    return logs

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = torch.load(args.data_path, map_location="cpu")
    dataset = dataset.to(device)
    
    # Setup dimensions
    input_dim = dataset.x.size(1)
    labels = len(dataset.label_names)
    hidden_dim = 128
    
    print(f"Dataset info:")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of labels: {labels}")
    print(f"Input feature dimension: {input_dim}")
    
    # Setup model and training
    model = setup_model(args.model_name, input_dim, hidden_dim, labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Start training
    training(args, model, dataset, optimizer, criterion)

if __name__ == "__main__":
    main()