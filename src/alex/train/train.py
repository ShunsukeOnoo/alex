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

    
TODO: 
- Check processor compatibility with the dataset (data type and format)
- Add them to processor config and use config values if not provided
  - padding
  - padding_side
  - max_length
- Add LoRA
- Assign learning rate for each modules
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


def set_trainable_parameters(model, freeze_vision: bool):
    """
    Set trainable parameters for the model.

    Args:
        model: The model to set trainable parameters.
        freeze_vision (bool): Whether to freeze the vision model.
    """
    if freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    deepspeed.init_distributed()

    model, processor = load_model_and_preprocessor(config)
    dataset = YouTubeDataset(transform=processor, **config['dataset'])

    set_trainable_parameters(model, config['freeze_vision'])    
    trainer = Trainer(
        model=model, train_dataset=dataset,
        args=TrainingArguments(**config["training"]),
        collate_fn=processor.collate_fn
    )
    result = trainer.train()

    save_path = os.path.join(config['training']['output_dir'], 'final_checkpoint')
    trainer.save_model(save_path)

if __name__ == "__main__":
    fire.Fire(main)