import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCNBackbone, SageBackbone
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
import time
import os
import json
from datetime import datetime
import argparse

def setup_directories(logs_path):
    """Create necessary directories for logs and models"""
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.dirname(logs_path), exist_ok=True)

def get_model(model_name, input_dim, device):
    """Initialize model based on model type"""
    hidden_dim = 128
    
    if model_name == 'gcn':
        model_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == "sage":
        model_backbone = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: gcn, sage")
    
    return model_backbone.to(device)

def load_and_process_data(data_path, device):
    """Load and process the dataset"""
    dataset = torch.load(data_path, map_location="cpu")
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(dataset)
    
    return (dataset, 
            train_data.to(device), 
            val_data.to(device), 
            test_data.to(device))

def train(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Positive edges
    pos_pred = model.decode(out, data.edge_label_index[:, data.edge_label == 1])

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_label_index[:, data.edge_label == 1],
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label[data.edge_label == 1].size(0)
    )
    neg_pred = model.decode(out, neg_edge_index)
    # Compute loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones(pos_pred.size(0), device=device))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros(neg_pred.size(0), device=device))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item(), torch.cat([pos_pred, neg_pred], dim=0)

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data)

        # Positive predictions
        pos_pred = model.decode(out, data.edge_label_index[:, data.edge_label == 1])
        pos_pred = torch.sigmoid(pos_pred)

        # Negative predictions
        neg_pred = model.decode(out, data.edge_label_index[:, data.edge_label == 0])
        neg_pred = torch.sigmoid(neg_pred)

        # Labels
        pos_labels = torch.ones(pos_pred.size(0))
        neg_labels = torch.zeros(neg_pred.size(0))
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # AUC score
        auc = roc_auc_score(labels.cpu(), preds.cpu())
        
        # Compute loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, pos_labels.to(device))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, neg_labels.to(device))
        loss = pos_loss + neg_loss
    
    return auc, loss.item()

def train_model(model, train_data, val_data, test_data, optimizer, dataset_name, model_name, logs_path, model_path):
    """Main training loop function"""
    device = next(model.parameters()).device
    num_epochs = 500
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logs = {
        'model_name': f"{dataset_name}_{model_name}_LP",
        'epochs': [],
        'train_auc': [],
        'val_auc': [],
        'test_auc': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'time_per_epoch': [],
        'best_epoch': 0,
        'best_val_auc': 0,
        'best_test_auc': 0,
        'total_training_time': 0
    }
    
    best_val_auc = 0
    total_time = 0
    
    print("\nStarting training for Link Prediction")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, _ = train(model, train_data, optimizer, device)
        train_auc = 0  # Placeholder as per existing code
        val_auc, val_loss = evaluate(model, val_data, device)
        test_auc, test_loss = evaluate(model, test_data, device)
        
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        # Update logs dictionary
        logs['epochs'].append(epoch)
        logs['train_auc'].append(train_auc)
        logs['val_auc'].append(val_auc)
        logs['test_auc'].append(test_auc)
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['test_loss'].append(test_loss)
        logs['time_per_epoch'].append(epoch_time)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            logs['best_epoch'] = epoch
            logs['best_val_auc'] = val_auc
            logs['best_test_auc'] = test_auc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'test_auc': test_auc
            }, model_path)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs-1}:")
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Print best accuracies every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"\nBest so far:")
            print(f"Val AUC: {best_val_auc:.4f}, Test AUC: {logs['best_test_auc']:.4f} (Epoch {logs['best_epoch']})")
            print(f"Average epoch time: {total_time/(epoch+1):.2f}s")
            print("-" * 50)
    
    logs['total_training_time'] = total_time
    log_file = os.path.join(logs_path, f'{dataset_name}_{model_name}_LP_{timestamp}.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return logs

def main():
    parser = argparse.ArgumentParser(description='Train Link Prediction Model')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
    parser.add_argument('--model_name', required=True, choices=['gcn', 'sage'], help='Model architecture')
    parser.add_argument('--data_path', required=True, help='Path to the dataset')
    parser.add_argument('--logs_path', required=True, help='Path to save logs')
    parser.add_argument('--model_path', required=True, help='Path to save model')
    args = parser.parse_args()

    setup_directories(args.logs_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and process data
    dataset, train_data, val_data, test_data = load_and_process_data(args.data_path, device)
    
    # Initialize model
    input_dim = dataset.x.size(1)
    model = get_model(args.model_name, input_dim, device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Train model
    logs = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        optimizer=optimizer,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        logs_path=args.logs_path,
        model_path=args.model_path
    )
    
    # Print final summary
    print("\nTraining completed!")
    print(f"Best val AUC: {logs['best_val_auc']:.4f} (Epoch {logs['best_epoch']})")
    print(f"Test AUC at best val: {logs['best_test_auc']:.4f}")
    print(f"Total training time: {logs['total_training_time']:.2f}s")
    print(f"Average epoch time: {logs['total_training_time']/500:.2f}s")

if __name__ == "__main__":
    main()
