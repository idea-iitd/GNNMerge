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
            backbone =  SageBackboneBackbone(input_dim, hidden_dim).to(device)
            mlp = GNNMLP(hidden_dim,labels).to(device)
            model_backbones.append(backbone)
            model_mlps.append(mlp)
        merged_backbone = SageBackboneBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: gcn, sage")
    
    return model_backbones, model_mlps, merged_backbone

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

def train(model_outputs, model_inputs, merged_backbone, optimizer, criterion):
    merged_backbone.train()
    optimizer.zero_grad()
    num_models = len(model_outputs)
    losses = [0] * num_models

    for layer_name in model_outputs[0].keys():
        for j in range(num_models):
            model_out = model_outputs[j][layer_name]
            model_inp = model_inputs[j][layer_name]
            layer = getattr(merged_backbone, layer_name, None)
            out = layer(model_inp[0], model_inp[1])
            losses[j] += criterion(out,model_out)
    loss = 0
    for losss in losses:
        loss+=losss
    loss.backward()
    optimizer.step()
    return losses

def merge_model(models, merged_backbone, datasets, num_layers, dataset_names, logs_path, model_name, logs_path):
    num_epochs = 1000
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
        logs[f'train_loss{i+1}'] = []

    total_time = 0

    num_models = len(models)  # Number of models
    model_outputs = {f"model{i+1}_outputs": {} for i in range(num_models)}
    model_inputs = {f"model{i+1}_inputs": {} for i in range(num_models)}
    model_hooks = {f"model{i+1}_hooks": [] for i in range(num_models)}
    model_mlps = []

    for j in range(num_models):
        model_mlps.append(models[j].mlp)

    for j in range(num_models):
        for i in range(1, num_layers+1):
            layer_name = f"conv{i}"
            layer = getattr(models[j].backbone, layer_name, None)
            model_hooks[j].append(layer.register_forward_hook(partial(hook_fn, outs=model_outputs[j], ins = model_inputs[j],layer_name=layer_name)))
    
    #Single forward pass to compute the inputs and outputs of merged models
    for j in range(num_models):
        _ = models[j].backbone(datasets[j])
    

    optimizer = torch.optim.Adam(merged_backbone.parameters(), lr=5e-2, betas=(0.9, 0.999), weight_decay=0.)
    criterion = nn.MSELoss(reduction='mean')

    print(f"\nStarting Merging Model")
    print("-" * 50)

    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_losses = train(model_outputs, model_inputs, merged_backbone, optimizer, criterion)
    
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time

        #Evaluate
        train_accs, val_accs, test_accs = evaluate(merged_backbone,model_mlps, datasets)
        
    

        logs['epochs'].append(epoch)
        for i in range(num_models):
            logs[f'train_acc{i+1}'].append(train_acc[i].item())
            logs[f'val_acc{i+1}'].append(val_acc[i].item())
            logs[f'test_acc{i+1}'].append(test_acc[i].item())
            logs[f'train_loss{i+1}'].append(train_loss[i].item())
        logs['time_per_epoch'].append(epoch_time)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/1000:")
            for i in range(len(models)):
                print(f"Dataset{i+1} - Train: {train_acc[i]:.4f}, Val: {val_acc[i]:.4f}, Test: {test_acc[i]:.4f}")
            print(f"Average epoch time: {total_time/(epoch+1):.2f}s")
            print("-" * 50)
        
    logs['total_training_time'] = total_time
    
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
            parma.requires_grad = False
    
    # Perform merging
    logs = merge_model(
        models = models,  merged_backbone=merged_backbone,
        datasets=datasets, num_layers=2,
        dataset_names=args.dataset_names, model_name=args.model_name,
        logs_path=args.logs_path
    )

if __name__ == "__main__":
    main()
