"""Train Alex model with DeepSpeed and HuggingFace Trainer.

Usage:
    python src/alex/train/train.py /path/to/config.yaml

Args:
    config_path (str): Path to the config file.

Config:
    training (dict): Arguments for the HuggingFace Trainer.

    pretrain_name (str): Name of the pretrained name for the langauge model.
    vision_pretrain_name (str): Name of the pretrained name for the vision model.
    model (dict): Additional arguments for the Alex model.
    preprocessor (dict): Arguments for the preprocessor.

    dataset (dict): Arguments for the YouTubeDataset.

    wandb (dict): Arguments for the Weights and Biases.


"""
from typing import Any, Dict, List, Tuple, Union
import os

import yaml
import fire
import torch
import deepspeed
from transformers import TrainingArguments, Trainer
from alex.model.factory import load_model_and_preprocessor
from alex.dataset.dataset import YouTubeDataset


def main(config_path: str):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    deepspeed.init_distributed()

    # TODO: Check processor compatibility with the dataset
    # TODO: Where should we perform preprocessing and padding? on dataset or on collate_fn?
    model, processor = load_model_and_preprocessor(config)
    dataset = YouTubeDataset(transform=processor, **config['dataset'])

    # TODO: Add LoRA
    # TODO: set trainable parameters
    # TODO: Set learning rate for each module
    model.train()

    # Set wandb project and name


    trainer = Trainer(
        model=model, train_dataset=dataset,
        args=TrainingArguments(**config["training"]),
    )

    with torch.autocast("cuda"):
        result = trainer.train()

    save_path = os.path.join(config['training']['output_dir'], 'final_ckeckpoint')
    trainer.save_model(save_path)

if __name__ == "__main__":
    fire.Fire(main)