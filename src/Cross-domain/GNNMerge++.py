import torch
import torch.nn as nn
from models import GCNBackbone, GNNMLP, GNNComplete, SageBackbone
from functools import partial
import gc
import sys
import time
import os
import json
import argparse
from datetime import datetime
import math
def parse_args():
    parser = argparse.ArgumentParser(description='Merge label split models')
    parser.add_argument(
        "--data_paths", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of dataset paths"
    )
    parser.add_argument(
        "--dataset_names", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of dataset names"
    )
    parser.add_argument(
        "--model_paths", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of model paths"
    )
    parser.add_argument('--model_name', type=str, required=True, help='Model architecture (gcn or sage)')
    parser.add_argument('--logs_path', type=str, required=True, help='Path to save logs')
    return parser.parse_args()

def init_models(datasets, model_name, device, hidden_dim=128):
    model_backbones = []
    model_mlps = []
    if model_name == 'gcn':
        for dataset in datasets:
            input_dim = dataset.x.size(1)
            labels = len(dataset.label_names)
            backbone =  GCNBackbone(input_dim, hidden_dim).to(device)
            mlp = GNNMLP(hidden_dim,labels).to(device)
            model_backbones.append(backbone)
            model_mlps.append(mlp)
        merged_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == "sage":
        for dataset in datasets:
            input_dim = dataset.x.size(1)
            labels = len(dataset.label_names)
            backbone =  SageBackbone(input_dim, hidden_dim).to(device)
            mlp = GNNMLP(hidden_dim,labels).to(device)
            model_backbones.append(backbone)
            model_mlps.append(mlp)
        merged_backbone = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: gcn, sage")
    
    return model_backbones, model_mlps, merged_backbone


def hook_fn(module, input, output, outs, ins, layer_name):
        outs[layer_name] = output
        ins[layer_name] = input

def accuracy(pred, labels, mask):
    correct = (pred[mask]==labels[mask]).sum()
    samples = mask.sum()
    return correct/samples

def evaluate(merged_backbone,model_mlps, datasets):
    merged_backbone.eval()
    train_accs = []
    val_accs = []
    test_accs = []
    for j in range(len(datasets)):
        dataset = datasets[j]
        out = merged_backbone(dataset)
        pred = model_mlps[j](out).argmax(dim = 1)
        train_acc = accuracy(pred, dataset.y, dataset.train_masks[0])
        val_acc = accuracy(pred, dataset.y, dataset.val_masks[0])
        test_acc = accuracy(pred, dataset.y, dataset.test_masks[0])
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    return train_accs, val_accs, test_accs

def least_squares(X,Y): 
    X = torch.cat(X, dim = 0)
    Y = torch.cat(Y, dim = 0)
    # Compute XtX and XtY
    XtX = X.T @ X
    XtY = X.T @ Y
    
    # Add regularization
    epsilon = 1e-6
    n = XtX.size(0)
    reg_matrix = epsilon * torch.eye(n, device=XtX.device)
    XtX_reg = XtX + reg_matrix
    try:
        W = torch.linalg.solve(XtX_reg, XtY)
    except RuntimeError:
        try:
            XtX_reg = XtX + (epsilon * 100) * torch.eye(n, device=XtX.device)
            W = torch.linalg.solve(XtX_reg, XtY)
        except RuntimeError:
            print("Warning: Using pseudo-inverse as matrix is still singular")
            W = torch.linalg.pinv(XtX_reg) @ XtY
    return W

def solve_lsq_gnn(model_outputs, model_inputs, merged_backbone, datasets, num_layers):
    for layer_name in model_outputs[0].keys():
        inputs = []
        targets = []
        for j in range(len(model_outputs)):
            output = model_outputs[j][layer_name]
            inp = model_inputs[j][layer_name][0]
            n = inp.shape[0]
            inp = inp/math.sqrt(n)
            output = output/math.sqrt(n)
            inputs.append(inp)
            targets.append(output)
        W = least_squares(inputs, targets)
        with torch.no_grad():
            layer = getattr(merged_backbone, layer_name[:5], None)
            layer.lin.weight.copy_(W.T)
            
def solve_least_sq(model_outputs, model_inputs, merged_backbone, datasets, num_layers, model_name):
    solve_lsq_gnn(model_outputs, model_inputs, merged_backbone, datasets, num_layers)

def merge_model(models, merged_backbone, datasets, num_layers, dataset_names, model_name, logs_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = {
        'epochs': [],
        'time_per_epoch': [],
        'total_training_time': 0
    }
    for i in range(len(models)):
        logs[f'train_acc{i+1}'] = []
        logs[f'val_acc{i+1}'] = []
        logs[f'test_acc{i+1}'] = []

    total_time = 0

    num_models = len(models)  # Number of models
    model_inputs = [{} for _ in range(num_models)]
    model_outputs = [{} for _ in range(num_models)]
    model_hooks = [[] for i in range(num_models)]

    model_mlps = []

    for j in range(num_models):
        model_mlps.append(models[j].mlp)

    for j in range(num_models):
        for i in range(1, num_layers+1):
            layer_name = f"conv{i}"
            layer_name1 = layer_name+".lin"
            layer = getattr(models[j].backbone, layer_name, None)
            model_hooks[j].append(layer.register_forward_hook(partial(hook_fn, outs=model_outputs[j], ins = model_inputs[j],layer_name=layer_name1)))
    
    #Single forward pass to compute the inputs and outputs of merged models
    for j in range(num_models):
        _ = models[j].backbone(datasets[j])
    
    print(f"\nStarting Merging Model")
    print("-" * 50)

    solve_least_sq(model_outputs, model_inputs, merged_backbone, datasets, num_layers, model_name)
    
    train_accs, val_accs, test_accs = evaluate(merged_backbone,model_mlps, datasets)
    for i in range(num_models):
        logs[f'train_acc{i+1}'].append(train_accs[i].item())
        logs[f'val_acc{i+1}'].append(val_accs[i].item())
        logs[f'test_acc{i+1}'].append(test_accs[i].item())
    for i in range(len(models)):
        print(f"Dataset{i+1} - Train: {train_accs[i]:.4f}, Val: {val_accs[i]:.4f}, Test: {test_accs[i]:.4f}")

    # Save logs
    os.makedirs(logs_path, exist_ok=True)
    dataset_part = "_".join(dataset_names)
    log_file = os.path.join(logs_path, f'{dataset_part}_{model_name}_{timestamp}.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return logs

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    datasets = []
    for data_path in args.data_paths:
        dataset = torch.load(data_path, map_location="cpu")
        dataset = dataset.to(device)
        datasets.append(dataset)
    
    # Initialize models
    model_backbones, model_mlps, merged_backbone = init_models(datasets, args.model_name, device)
    
    models = []
    for i in range(len(model_backbones)):
        models.append(GNNComplete(model_backbones[i],model_mlps[i]).to(device))

    for i in range(len(models)):
        models[i].load_state_dict(torch.load(args.model_paths[i])['model_state_dict'])
        for param in models[i].parameters():
            param.requires_grad = False
    
    # Perform merging
    logs = merge_model(
        models = models,  merged_backbone=merged_backbone,
        datasets=datasets, num_layers=2,
        dataset_names=args.dataset_names, model_name=args.model_name,
        logs_path=args.logs_path
    )

if __name__ == "__main__":
    main()
