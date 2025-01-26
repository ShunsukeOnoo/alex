"""
Instantiate the model.
"""
from typing import Any, Dict, List, Tuple, Union

import torch

from transformers import CLIPVisionModel, AutoTokenizer, CLIPProcessor
from .modeling_alex_opt import AlexOPTForAction, AlexConfig, AlexVisionConfig
from .processing_alex import AlexProcessor


def load_model_and_preprocessor(config: Dict[str, Any]):
    """
    Load model and preprocessor from the config.

    Contents of the config:
    
    """
    pretrain_name = config['pretrain_name']
    vision_pretrain_name = config['vision_pretrain_name']

    # load preprocessor first
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    # TODO: This is not the vision processor
    vision_processor = CLIPProcessor.from_pretrained(vision_pretrain_name)
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
        # TODO: Add processor these attributes
        frame_token_id=processor.frame_token_id,
        frame_end_token_id=processor.frame_end_token_id,
        **config['model']
    )

    # instantiate the model and load weights for the language model
    model = AlexOPTForAction.from_pretrained(pretrain_name, config=config)
    # load weights for the vision model
    # This is double initialization but it is necessary to load the weights
    model.vision_model.model = CLIPVisionModel.from_pretrained(vision_pretrain_name)

    # Expand the embedding layer for the language model
    model.resize_token_embeddings(len(processor.tokenizer))  # TODO: Is this correct?
    return model, processor