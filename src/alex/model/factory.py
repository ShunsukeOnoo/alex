"""
Instantiate the model.
"""
from typing import Any, Dict, List, Tuple, Union

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from .modeling_alex_opt import AlexOPTForAction, AlexConfig, AlexVisionConfig
from .processing_alex import AlexProcessor


def load_model_and_preprocessor(config: Dict[str, Any]):
    """
    Load model and preprocessor from the config for training the model.

    Contents of the config:
        - pretrain_name: The name of the pre-trained model for the langauge model.
        - vision_pretrain_name: The name of the pre-trained model for the vision model.
        - preprocessor: The configuration for the preprocessor.
        - model: The configuration for the model.
    """
    pretrain_name = config['pretrain_name']
    vision_pretrain_name = config['vision_pretrain_name']

    # load preprocessor first
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    vision_processor = CLIPImageProcessor.from_pretrained(vision_pretrain_name)
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
    model.resize_token_embeddings(len(processor.tokenizer))
    return model, processor