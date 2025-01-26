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
from transformers import Trainer, TrainingArguments, CLIPVisionModel, AutoTokenizer
from alex.model.modeling_alex_opt import AlexOPTForAction, AlexConfig, AlexVisionConfig
from alex.model.processing_alex import AlexProcessor
from alex.dataset.dataset import YouTubeDataset


def load_model_and_preprocessor(config: Dict[str, Any]):
    pretrain_name = config['pretrain_name']
    vision_pretrain_name = config['vision_pretrain_name']

    # load preprocessor first
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    vision_processor = CLIPVisionProcessor.from_pretrained(vision_pretrain_name)
    processor = AlexProcessor(
        tokenizer=tokenizer, 
        vision_processor=vision_processor,
        **config['preprocessor']
    )
    
    # prepare config for the model
    vision_config = AlexVisionConfig.from_pretrained(vision_pretrain_name)

    config = AlexConfig.from_pretrained(pretrain_name)
    config.add_config(
        vision_config=vision_config, 
        # TODO: Processor does not have these attributes
        frame_token_id=processor.frame_token_id,
        frame_end_token_id=processor.frame_end_token_id,
        **config['model']
    )

    # load model and its weights
    model = AlexOPTForAction.from_pretrained(pretrain_name, config=config)
    # This is double initialization but it is necessary to load the weights
    model.vision_model.model = CLIPVisionModel.from_pretrained(vision_pretrain_name)

    # Expand the embedding layer for the language model
    model.resize_token_embeddings(len(processor.tokenizer))  # TODO: Is this correct?
    return model, processor


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