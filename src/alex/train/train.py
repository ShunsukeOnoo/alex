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
import os
import json
import warnings

import wandb
import yaml
import fire
import torch
import deepspeed
from transformers import TrainingArguments, Trainer, TrainerCallback
from alex.model.factory import load_model_and_preprocessor
from alex.model.processing_alex import PaddingCollator
from alex.dataset.dataset import YouTubeDataset, PreprocessedDataset


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


class LossLoggingTrainer(Trainer):
    """
    Trainer class that reports the action_loss and lm_loss to wandb.

    TODO: This implementation relies on return_dict=True in the model.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        # Extract losses
        loss = outputs["loss"]
        action_loss = outputs["action_loss"]
        lm_loss = outputs["lm_loss"]

        # Log additional losses using self.log() to sync with Trainer's logging
        self.log({"action_loss": action_loss.item(), "lm_loss": lm_loss.item()})

        return (loss, outputs) if return_outputs else loss


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # trainer handles the wandb.init. instead set environment variables
    os.environ["WANDB_PROJECT"] = config['wandb']['project']
    os.environ['WANDB_RUN_NAME'] = config['wandb']['name']

    deepspeed.init_distributed()

    model, processor = load_model_and_preprocessor(config)

    if 'preprocessed_data_dir' in config:
        # use preprocessed data
        # make sure the processor is compatible with the model
        warnings.warn("Using preprocessed data. Make sure the processor is compatible with the model.")
        dataset = PreprocessedDataset(config['preprocessed_data_dir'])
    else:
        # use raw data and preprocess on the fly
        dataset = YouTubeDataset(transform=processor, **config['dataset'])
    
    collator = PaddingCollator(
        max_length=config['max_length'], 
        pad_token_id=processor.tokenizer.pad_token_id, 
    )

    set_trainable_parameters(model, config['freeze_vision'])    
    trainer = LossLoggingTrainer(
        model=model, train_dataset=dataset,
        args=TrainingArguments(**config["training"]),
        data_collator=collator,
    )
    result = trainer.train()

    # is this correct way to save the model?
    save_path = os.path.join(config['training']['output_dir'], 'final_checkpoint')
    trainer.save_model(save_path)

    # also save the tokenizer and processor

if __name__ == "__main__":
    fire.Fire(main)