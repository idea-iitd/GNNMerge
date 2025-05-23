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
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Model architecture (gcn or sage)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model1_path', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2_path', type=str, required=True, help='Path to second model')
    parser.add_argument('--logs_path', type=str, required=True, help='Path to save logs')
    return parser.parse_args()

def init_models(dataset, model_name, device, hidden_dim=128):
    input_dim = dataset.x.size(1)
    labels = len(dataset.label_names)
    
    if model_name == 'gcn':
        model1_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        model2_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        model3_backbone = GCNBackbone(input_dim, hidden_dim).to(device)
        model1_mlp = GNNMLP(hidden_dim, (labels+1)//2).to(device)
        model2_mlp = GNNMLP(hidden_dim, labels//2).to(device)
    elif model_name == "sage":
        model1_backbone = SageBackbone(input_dim, hidden_dim).to(device)
        model2_backbone = SageBackbone(input_dim, hidden_dim).to(device)
        model3_backbone = SageBackbone(input_dim, hidden_dim).to(device)
        model1_mlp = GNNMLP(hidden_dim, (labels+1)//2).to(device)
        model2_mlp = GNNMLP(hidden_dim, labels//2).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: gcn, sage")
    
    return model1_backbone, model2_backbone, model3_backbone, model1_mlp, model2_mlp

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

def evaluate(model3_backbone, mlp1, mlp2, dataset, train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2):
    model3_backbone.eval()
    out = model3_backbone(dataset)
    pred1 = mlp1(out).argmax(dim = 1)
    pred2 = mlp2(out).argmax(dim = 1)
    train_acc1 = accuracy(pred1, dataset.y, train_mask1)
    train_acc2 = accuracy(pred2, dataset.y, train_mask2)
    val_acc1 = accuracy(pred1, dataset.y, val_mask1)
    val_acc2 = accuracy(pred2, dataset.y, val_mask2)
    test_acc1 = accuracy(pred1, dataset.y, test_mask1)
    test_acc2 = accuracy(pred2, dataset.y, test_mask2)
    return train_acc1, train_acc2, val_acc1, val_acc2, test_acc1, test_acc2

def train(model1_outputs, model1_inputs, model2_outputs, model2_inputs, model3_backbone, train_mask1, train_mask2, optimizer, criterion):
    model3_backbone.train()
    optimizer.zero_grad()
    loss1 = 0
    loss2 = 0
    for layer_name in model1_outputs.keys():
        model1_out = model1_outputs[layer_name]
        model1_inp = model1_inputs[layer_name]
        model2_out = model2_outputs[layer_name]
        model2_inp = model2_inputs[layer_name]
        layer = getattr(model3_backbone, layer_name, None)
        out = layer(model1_inp[0], model1_inp[1])
        loss1 += criterion(out[train_mask1],model1_out[train_mask1])
        out = layer(model2_inp[0], model2_inp[1])
        loss2 += criterion(out[train_mask2],model2_out[train_mask2])
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    return loss1, loss2

def merge_model(model1, model2, model3_backbone, dataset, num_layers, 
                train_mask1, train_mask2, val_mask1, val_mask2, 
                test_mask1, test_mask2, dataset_name, model_name, logs_path):
    num_epochs = 1000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = {
        'model_name': f"{dataset_name}_{model_name}",
        'epochs': [],
        'train_acc1': [],
        'val_acc1': [],
        'test_acc1': [],
        'train_acc2': [],
        'val_acc2': [],
        'test_acc2': [],
        'train_loss1': [],
        'train_loss2': [],
        'normalized_train' : [],
        'normalized_val' : [],
        'normalized_test' : [],
        'time_per_epoch': [],
        'total_training_time': 0
    }
    total_time = 0

    model1_outputs = {}
    model1_inputs = {}
    model2_outputs = {}
    model2_inputs = {}
    model1_hooks = []
    model2_hooks = []

    
    for i in range(1,num_layers+1):
        layer_name = f"conv{i}"
        layer = getattr(model1.backbone, layer_name, None)
        model1_hooks.append(layer.register_forward_hook(partial(hook_fn, outs=model1_outputs, ins = model1_inputs,layer_name=layer_name)))
        layer = getattr(model2.backbone, layer_name, None)
        model2_hooks.append(layer.register_forward_hook(partial(hook_fn, outs=model2_outputs, ins = model2_inputs,layer_name=layer_name)))

    #Single forward pass to compute the inputs and outputs of merged models    
    _  = model1.backbone(dataset)
    _  = model2.backbone(dataset)

    optimizer = torch.optim.Adam(model3_backbone.parameters(), lr=5e-2, betas=(0.9, 0.999), weight_decay=0.)
    criterion = nn.MSELoss(reduction='mean')

    print(f"\nStarting Merging Model")
    print("-" * 50)

    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss1, train_loss2 = train(model1_outputs, model1_inputs, model2_outputs, model2_inputs, model3_backbone, train_mask1, train_mask2, optimizer, criterion)
        
    
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time

        #Evaluate
        train_acc1, train_acc2, val_acc1, val_acc2, test_acc1, test_acc2 = evaluate(model3_backbone, model1.mlp, model2.mlp, dataset, train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2)
        
        
        # Update logs
        normalized_train = (train_acc1*train_mask1.sum()+train_acc2*train_mask2.sum())/(train_mask1.sum()+train_mask2.sum())
        normalized_val = (val_acc1*val_mask1.sum()+val_acc2*val_mask2.sum())/(val_mask1.sum()+val_mask2.sum())
        normalized_test = (test_acc1*test_mask1.sum()+test_acc2*test_mask2.sum())/(test_mask1.sum()+test_mask2.sum())

        logs['epochs'].append(epoch)
        logs['train_acc1'].append(train_acc1.item())
        logs['val_acc1'].append(val_acc1.item())
        logs['test_acc1'].append(test_acc1.item())
        logs['train_acc2'].append(train_acc2.item())
        logs['val_acc2'].append(val_acc2.item())
        logs['test_acc2'].append(test_acc2.item())
        logs['train_loss1'].append(train_loss1.item())
        logs['train_loss2'].append(train_loss2.item())
        logs['normalized_test'].append(normalized_test.item())
        logs['normalized_val'].append(normalized_val.item())
        logs['normalized_train'].append(normalized_train.item())

        logs['time_per_epoch'].append(epoch_time)
        
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs-1}:")
        print(f"Train1: {train_acc1:.4f}, Val: {val_acc1:.4f}, Test: {test_acc1:.4f}")
        print(f"Train2: {train_acc2:.4f}, Val: {val_acc2:.4f}, Test: {test_acc2:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Print best accuracies every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Average epoch time: {total_time/(epoch+1):.2f}s")
            print("-" * 50)
    
    logs['total_training_time'] = total_time
    
    # Save logs
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f'{dataset_name}_{model_name}_{timestamp}.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return logs

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = torch.load(args.data_path, map_location="cpu")
    dataset = dataset.to(device)
    
    # Initialize models
    model1_backbone, model2_backbone, model3_backbone, model1_mlp, model2_mlp = init_models(
        dataset, args.model_name, device
    )
    
    # Create complete models
    model1 = GNNComplete(model1_backbone, model1_mlp).to(device)
    model2 = GNNComplete(model2_backbone, model2_mlp).to(device)
    
    # Load model weights
    model1.load_state_dict(torch.load(args.model1_path)['model_state_dict'])
    model2.load_state_dict(torch.load(args.model2_path)['model_state_dict'])
    
    # Freeze parameters
    for param in model1.parameters():
        param.requires_grad = False
    for param in model2.parameters():
        param.requires_grad = False
    
    # Create masks
    masks = create_masks(dataset)
    train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2 = [m.to(device) for m in masks[:6]]
    
    # Perform merging
    logs = merge_model(
        model1=model1, model2=model2, model3_backbone=model3_backbone,
        dataset=dataset, num_layers=2,
        train_mask1=train_mask1, train_mask2=train_mask2,
        val_mask1=val_mask1, val_mask2=val_mask2,
        test_mask1=test_mask1, test_mask2=test_mask2,
        dataset_name=args.dataset_name, model_name=args.model_name,
        logs_path=args.logs_path
    )

if __name__ == "__main__":
    main()
