import torch
import torch.nn as nn
from models import GCNBackbone, GNNMLP, GNNComplete, SageBackbone
from functools import partial
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import gc
import sys
import time
import os
import json
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Merge NC and LP')
    parser.add_argument('--nc_dataset_name', type=str, required=True, help='Name of the dataset for node classification')
    parser.add_argument('--lp_dataset_name', type=str, required=True, help='Name of the dataset for link prediction')
    parser.add_argument('--model_name', type=str, required=True, help='Model architecture (gcn or sage)')
    parser.add_argument('--nc_data_path', type=str, required=True, help='Path to dataset for node classification')
    parser.add_argument('--lp_data_path', type=str, required=True, help='Path to dataset for link prediction')

    parser.add_argument('--nc_model_path', type=str, required=True, help='Path to node classification')
    parser.add_argument('--lp_model_path', type=str, required=True, help='Path to link prediction')
    parser.add_argument('--logs_path', type=str, required=True, help='Path to save logs')
    return parser.parse_args()

def init_models(nc_dataset, lp_dataset, model_name, device, hidden_dim=128):

    input_dim = nc_dataset.x.size(1)
    labels = len(nc_dataset.label_names)
    
    if model_name == 'gcn':
        nc_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        nc_mlp = GNNMLP(hidden_dim, labels).to(device)
        lp_model = GCNBackbone(input_dim, hidden_dim).to(device)
        nc_model = GNNComplete(nc_backbone, nc_mlp).to(device)
        model3_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        
    elif model_name == "sage":
        nc_backbone = SageBackbone(input_dim, hidden_dim).to(device)
        nc_mlp = GNNMLP(hidden_dim, labels).to(device)
        lp_model = SageBackbone(input_dim, hidden_dim).to(device)
        nc_model = GNNComplete(nc_backbone, nc_mlp).to(device)
        model3_backbone = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: gcn, sage")
    
    return nc_model, lp_model, model3_backbone

def create_masks(dataset):
    N = dataset.num_nodes
    labels = len(dataset.label_names)
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
    
    # Get labels for these indices
    train_labels = dataset.y[train_indices]
    val_labels = dataset.y[val_indices]
    test_labels = dataset.y[test_indices]

    # Create a copy of the original labels before modifying
    original_y = dataset.y.clone()

    # Assign data points to respective masks based on their labels
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

    # Create a mapping for second model label
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(list(classes_set2)))}
    # Adjust labels for second model using the mapping
    dataset.y = original_y.clone()  # Reset to original labels
    for idx in range(len(dataset.y)):
        if (train_mask2[idx] or test_mask2[idx] or val_mask2[idx]):
            dataset.y[idx] = label_mapping[dataset.y[idx].item()]

    return (train_mask1, train_mask2, val_mask1, val_mask2, 
            test_mask1, test_mask2, classes_set1, classes_set2)

def hook_fn(module, input, output, outs, ins, layer_name):
        outs[layer_name] = output
        ins[layer_name] = input

def accuracy(pred, labels, mask):
    correct = (pred[mask]==labels[mask]).sum()
    samples = mask.sum()
    return correct/samples

def link_prediction_auc(model, data, edge_label_index, edge_label):
    model.eval()
    with torch.no_grad():
        out = model(data)
        
        # Positive predictions
        pos_pred = model.decode(out, edge_label_index[:, edge_label == 1])
        pos_pred = torch.sigmoid(pos_pred)
        
        # Negative predictions
        neg_pred = model.decode(out, edge_label_index[:, edge_label == 0])
        neg_pred = torch.sigmoid(neg_pred)
        
        # Labels
        pos_labels = torch.ones(pos_pred.size(0))
        neg_labels = torch.zeros(neg_pred.size(0))
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # AUC score
        auc = roc_auc_score(labels.cpu(), preds.cpu())
    return auc

def evaluate(merged_backbone, nc_mlp, nc_dataset, lp_val_data, lp_test_data):
    merged_backbone.eval()
    # Evaluate Node Classification on dataset
    out_nc = merged_backbone(nc_dataset)
    pred = nc_mlp(out_nc).argmax(dim=1)
    train_acc = accuracy(pred, nc_dataset.y, nc_dataset.train_masks[0])
    val_acc = accuracy(pred, nc_dataset.y, nc_dataset.val_masks[0])
    test_acc = accuracy(pred, nc_dataset.y, nc_dataset.test_masks[0])
    
    # Evaluate Link Prediction on dataset2
    val_auc = link_prediction_auc(
        merged_backbone,
        lp_val_data,
        lp_val_data.edge_label_index,
        lp_val_data.edge_label
    )
    test_auc = link_prediction_auc(
        merged_backbone,
        lp_test_data,
        lp_test_data.edge_label_index,
        lp_test_data.edge_label
    )
    
    return train_acc, val_acc, test_acc, val_auc, test_auc

def train(nc_outputs, nc_inputs, lp_outputs, lp_inputs, merged_backbone, optimizer, criterion):
    # return
    merged_backbone.train()
    optimizer.zero_grad()
    loss_nc = 0
    loss_lp = 0
    for layer_name in nc_outputs.keys():
        # Process node classification (dataset1)
        nc_out = nc_outputs[layer_name]
        nc_inp = nc_inputs[layer_name]
        layer = getattr(merged_backbone, layer_name, None)
        out_nc = layer(nc_inp[0], nc_inp[1])
        loss_nc += criterion(out_nc, nc_out)
        
        # Process link prediction (dataset2)
        lp_out = lp_outputs[layer_name]
        lp_inp = lp_inputs[layer_name]
        out_lp = layer(lp_inp[0], lp_inp[1])
        loss_lp += criterion(out_lp, lp_out)

    loss = loss_nc + loss_lp
    loss.backward()
    optimizer.step()
    return loss_nc, loss_lp


def merge_model(nc_model, lp_model, merged_backbone,
        nc_dataset, lp_dataset, lp_val_data, lp_test_data, num_layers, nc_dataset_name, lp_dataset_name, model_name,
        logs_path):
    num_epochs = 1000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = {
        'model_name': f"{nc_dataset_name}_{lp_dataset_name}_{model_name}_merged",
        'epochs': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'val_auc': [], 'test_auc': [],
        'train_loss_nc': [], 'train_loss_lp': [],
        'time_per_epoch': [],
        'total_training_time': 0
    }
    total_time = 0

    # Setup hooks to capture intermediate outputs
    nc_outputs = {}
    nc_inputs = {}
    lp_outputs = {}
    lp_inputs = {}
    nc_hooks = []
    lp_hooks = []
    
    for i in range(1, num_layers+1):
        layer_name = f"conv{i}"
        layer = getattr(nc_model.backbone, layer_name, None)
        nc_hooks.append(layer.register_forward_hook(
            partial(hook_fn, outs=nc_outputs, ins=nc_inputs, layer_name=layer_name)))
        layer = getattr(lp_model, layer_name, None)
        lp_hooks.append(layer.register_forward_hook(
            partial(hook_fn, outs=lp_outputs, ins=lp_inputs, layer_name=layer_name)))

    #Single forward pass to compute the inputs and outputs of merged models    
    _  = nc_model.backbone(dataset)
    _  = lp_model(dataset)

    optimizer = torch.optim.Adam(model3_backbone.parameters(), lr=5e-2, betas=(0.9, 0.999), weight_decay=0.)
    criterion = nn.MSELoss(reduction='mean')

    print(f"\nStarting Merging Model")
    print("-" * 50)

    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss_nc, train_loss_lp = train(
                nc_outputs, nc_inputs, 
                lp_outputs, lp_inputs,
                merged_backbone, optimizer, criterion)
        
    
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time

        #Evaluate
        train_acc, val_acc, test_acc, val_auc, test_auc = evaluate(
                merged_backbone, nc_model.mlp, nc_dataset, lp_val_data, lp_test_data
            )
        
        

        logs['epochs'].append(epoch)
        logs['train_acc'].append(float(train_acc))
        logs['val_acc'].append(float(val_acc))
        logs['test_acc'].append(float(test_acc))
        logs['val_auc'].append(float(val_auc))
        logs['test_auc'].append(float(test_auc))
        logs['train_loss_nc'].append(float(train_loss_nc))
        logs['train_loss_lp'].append(float(train_loss_lp))
        logs['time_per_epoch'].append(float(epoch_time))
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Node Classification - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
            print(f"Link Prediction - Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
            print("-" * 50)
        

    logs['total_training_time'] = total_time
    
    # Save logs
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f'{nc_dataset_name}_{lp_dataset_name}_{model_name}_{timestamp}.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return logs

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    nc_dataset = torch.load(args.nc_data_path, map_location="cpu")
    lp_dataset = torch.load(args.lp_data_path, map_location="cpu")
    nc_dataset = nc_dataset.to(device)
    lp_dataset = lp_dataset.to(device)
    
    # Initialize models
    nc_model, lp_model, model3_backbone = init_models(
        nc_dataset, lp_dataset, args.model_name, device
    )
    
    
    # Load model weights
    nc_model.load_state_dict(torch.load(args.nc_model_path)['model_state_dict'])
    lp_model.load_state_dict(torch.load(args.lp_model_path)['model_state_dict'])
    
    # Freeze parameters
    for param in nc_model.parameters():
        param.requires_grad = False
    for param in lp_model.parameters():
        param.requires_grad = False
    
    from torch_geometric.transforms import RandomLinkSplit
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
    lp_train_data, lp_val_data, lp_test_data = transform(lp_dataset)

    # Perform merging
    logs = merge_model(
        nc_model = nc_model, lp_model = lp_model, merged_backbone = model3_backbone
        nc_dataset=nc_dataset, lp_dataset = lp_train_data, lp_val_data = lp_val_data, lp_test_data = lp_test_data, num_layers=2,
        nc_dataset_name=args.nc_dataset_name, lp_dataset_name = args.lp_dataset_name, model_name=args.model_name,
        logs_path=args.logs_path
    )

if __name__ == "__main__":
    main()
