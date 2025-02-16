"""
Preprocess and save samples for training.

Usage:
    python preprocess_data.py --config_path <path_to_config_file>

In addition to the normal training config, the config file must contain the following keys:
    - preprocessed_data_dir (str): Directory to save preprocessed samples.


"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from alex.dataset.dataset import YouTubeDataset
from alex.model.factory import load_preprocessor


def save_preprocessed_dataset(dataset, processor, save_dir, num_workers=4):
    """
    Preprocess the dataset and save each sample as a .pt file.

    Args:
        dataset (Dataset): The dataset object (YouTubeDataset).
        processor (callable): The processor function to transform samples.
        save_dir (str): Directory to save preprocessed samples.
        num_workers (int): Number of workers for the dataloader.
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(len(dataset)), total=len(dataset), desc="Processing and saving dataset"):
        sample = dataset[idx]
        processed_sample = processor(**sample)

        save_path = os.path.join(save_dir, f"sample_{idx}.pt")
        torch.save(processed_sample, save_path)

    print(f"Preprocessed dataset saved in {save_dir}")


def main(config_path, num_workers):
    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert 'preprocessed_data_dir' in config, "Please provide preprocessed_data_dir in the config file."

    # load dataset and processor
    dataset = YouTubeDataset(transform=None, **config['dataset'])
    processor = load_preprocessor(config)

    save_preprocessed_dataset(
        dataset, 
        processor, 
        save_dir=config['preprocessed_data_dir'],
        num_workers=num_workers
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    args = parser.parse_args()
    main(args.config_path, args.num_workers)