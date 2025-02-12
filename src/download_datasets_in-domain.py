import os
import torch
import shutil
import argparse
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor, Reddit
import time
from torch_sparse import SparseTensor

def safe_download_dataset(download_func, dataset_name):
    """Safely download a dataset with error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\nAttempting to download {dataset_name} (Attempt {attempt + 1}/{max_retries})")
            return download_func()
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                # Clean up any partial downloads
                if os.path.exists('temp_download'):
                    shutil.rmtree('temp_download')
                os.makedirs("temp_download", exist_ok=True)
            else:
                print(f"Failed to download {dataset_name} after {max_retries} attempts")
                return None

def process_dataset(data, train_mask, val_mask, test_mask, num_classes):
    """Process any dataset into a consistent format"""
    processed_data = data
    
    print("\nInput mask types:")
    print(f"Train mask type: {type(train_mask)}, shape/len: {train_mask.shape if hasattr(train_mask, 'shape') else len(train_mask)}")
    print(f"Val mask type: {type(val_mask)}, shape/len: {val_mask.shape if hasattr(val_mask, 'shape') else len(val_mask)}")
    print(f"Test mask type: {type(test_mask)}, shape/len: {test_mask.shape if hasattr(test_mask, 'shape') else len(test_mask)}")
    
    # Convert masks to list of indices
    if isinstance(train_mask, torch.Tensor) and train_mask.dtype == torch.bool:
        # If masks are boolean tensors, convert to indices
        processed_data.train_masks = [train_mask.nonzero(as_tuple=False).view(-1)]
        processed_data.val_masks = [val_mask.nonzero(as_tuple=False).view(-1)]
        processed_data.test_masks = [test_mask.nonzero(as_tuple=False).view(-1)]
    elif isinstance(train_mask, torch.Tensor):
        # If masks are already indices
        processed_data.train_masks = [train_mask]
        processed_data.val_masks = [val_mask]
        processed_data.test_masks = [test_mask]
    else:
        # If masks are lists or other format, convert to tensor indices
        processed_data.train_masks = [torch.tensor(train_mask, dtype=torch.long)]
        processed_data.val_masks = [torch.tensor(val_mask, dtype=torch.long)]
        processed_data.test_masks = [torch.tensor(test_mask, dtype=torch.long)]
    
    # Add label names
    processed_data.label_names = [str(i) for i in range(num_classes)]
    
    # Ensure y is the right format
    if hasattr(processed_data, 'y') and processed_data.y is not None:
        processed_data.y = processed_data.y.squeeze()
        if processed_data.y.dim() > 1:
            processed_data.y = processed_data.y.argmax(dim=1)
    
    # Verify the format
    assert all(isinstance(mask, list) for mask in [processed_data.train_masks, processed_data.val_masks, processed_data.test_masks]), \
        "All masks should be lists"
    assert all(isinstance(mask, torch.Tensor) for mask in processed_data.train_masks + processed_data.val_masks + processed_data.test_masks), \
        "All mask elements should be tensors"
    assert all(mask.dtype == torch.long for mask in processed_data.train_masks + processed_data.val_masks + processed_data.test_masks), \
        "All mask elements should be long tensors (indices)"
    
    print("\nProcessed masks:")
    print("Train masks:")
    for i, mask in enumerate(processed_data.train_masks):
        print(f"Split {i}: shape={mask.shape}, dtype={mask.dtype}")
        print(f"First 5 indices: {mask[:5].tolist()}")
    
    print("\nVal masks:")
    for i, mask in enumerate(processed_data.val_masks):
        print(f"Split {i}: shape={mask.shape}, dtype={mask.dtype}")
        print(f"First 5 indices: {mask[:5].tolist()}")
    
    print("\nTest masks:")
    for i, mask in enumerate(processed_data.test_masks):
        print(f"Split {i}: shape={mask.shape}, dtype={mask.dtype}")
        print(f"First 5 indices: {mask[:5].tolist()}")
    
    print(f"\nProcessed masks format summary:")
    print(f"Train masks: {[mask.shape for mask in processed_data.train_masks]}")
    print(f"Val masks: {[mask.shape for mask in processed_data.val_masks]}")
    print(f"Test masks: {[mask.shape for mask in processed_data.test_masks]}")
    
    return processed_data

def download_ogbn_arxiv(save_path):
    print("\nDownloading OGBN-Arxiv dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='temp_download')
    data = dataset[0]

    num_nodes = data.num_nodes
    data.edge_index = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    data.edge_index=data.edge_index.to_symmetric()
    
    splits = dataset.get_idx_split()
    processed_data = process_dataset(
        data=data,
        train_mask=splits['train'],
        val_mask=splits['valid'],
        test_mask=splits['test'],
        num_classes=dataset.num_classes
    )
    
    output_path = os.path.join(save_path, 'ogbn_arxiv.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data

def download_planetoid(name, save_path):
    print(f"\nDownloading {name} dataset...")
    dataset = Planetoid(root='/tmp/'+name, name=name)
    data = dataset[0]

    num_nodes = data.num_nodes
    data.edge_index = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    data.edge_index=data.edge_index.to_symmetric()
    
    processed_data = process_dataset(
        data=data,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_classes=dataset.num_classes
    )
    
    output_path = os.path.join(save_path, f'{name.lower()}.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data

def download_wikics(save_path):
    print("\nDownloading WikiCS dataset...")
    dataset = WikiCS(root='temp_download')
    data = dataset[0]

    num_nodes = data.num_nodes
    data.edge_index = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    data.edge_index = data.edge_index.to_symmetric()

    processed_data = data
    
    train_masks = []
    val_masks = []
    test_mask = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    for split_idx in range(data.train_mask.size(1)):
        train_mask = data.train_mask[:, split_idx].nonzero(as_tuple=False).view(-1)
        val_mask = data.val_mask[:, split_idx].nonzero(as_tuple=False).view(-1)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
    
    processed_data.train_masks = train_masks
    processed_data.val_masks = val_masks
    processed_data.test_masks = [test_mask]
    processed_data.label_names = [str(i) for i in range(dataset.num_classes)]
    
    output_path = os.path.join(save_path, 'wikics.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data

def download_amazon(name, save_path):
    print(f"\nDownloading Amazon-{name} dataset...")
    dataset = Amazon(root='temp_download', name=name)
    data = dataset[0]

    num_nodes = data.num_nodes
    data.edge_index = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    data.edge_index = data.edge_index.to_symmetric()
    
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = indices[:train_size]
    val_mask = indices[train_size:train_size+val_size]
    test_mask = indices[train_size+val_size:]
    
    processed_data = process_dataset(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes
    )
    
    output_path = os.path.join(save_path, f'amazon_{name.lower()}.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data

def download_reddit(save_path):
    print("\nDownloading Reddit dataset...")
    dataset = Reddit(root='temp_download')
    data = dataset[0]

    num_nodes = data.num_nodes
    data.edge_index = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    data.edge_index = data.edge_index.to_symmetric()
    
    processed_data = process_dataset(
        data=data,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_classes=dataset.num_classes
    )
    
    output_path = os.path.join(save_path, 'reddit.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data

def print_dataset_info(data, name):
    print(f"\n{name} Dataset Info:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.nnz()}")
    print(f"Number of node features: {data.x.size(1)}")
    print(f"Number of classes: {len(data.label_names)}")
    print(f"Number of training nodes: {len(data.train_masks[0])}")
    print(f"Number of validation nodes: {len(data.val_masks[0])}")
    print(f"Number of test nodes: {len(data.test_masks[0])}")
    print(f"Label range: {data.y.min().item()} to {data.y.max().item()}")

def main():
    parser = argparse.ArgumentParser(description='Download and process graph datasets')
    parser.add_argument('--save_path', type=str, default='../datasets',
                        help='Path to save the processed datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ogbn-arxiv', 'cora', 'wikics', 'amazon-photo', 
                                'amazon-computers', 'reddit'],  # Added reddit
                        help='Name of the dataset to download')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs("temp_download", exist_ok=True)
    
    try:
        if args.dataset == 'ogbn-arxiv':
            data = safe_download_dataset(lambda: download_ogbn_arxiv(args.save_path), 'OGBN-Arxiv')
        elif args.dataset == 'cora':
            data = safe_download_dataset(lambda: download_planetoid('Cora', args.save_path), 'Cora')
        elif args.dataset == 'wikics':
            data = safe_download_dataset(lambda: download_wikics(args.save_path), 'WikiCS')
        elif args.dataset == 'amazon-photo':
            data = safe_download_dataset(lambda: download_amazon('Photo', args.save_path), 'Amazon-Photo')
        elif args.dataset == 'amazon-computers':
            data = safe_download_dataset(lambda: download_amazon('Computers', args.save_path), 'Amazon-Computers')
        elif args.dataset == 'reddit':  # Added reddit case
            data = safe_download_dataset(lambda: download_reddit(args.save_path), 'Reddit')
        
        if data is not None:
            print_dataset_info(data, args.dataset)
    
    finally:
        # Clean up
        if os.path.exists('temp_download'):
            shutil.rmtree('temp_download')

if __name__ == "__main__":
    main()