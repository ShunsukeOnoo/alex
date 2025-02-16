"""
Instantiate the model.
"""
from typing import Any, Dict, List, Tuple, Union
from torchvision import transforms
from transformers import CLIPVisionModel, AutoTokenizer
from .modeling_alex_opt import AlexOPTForAction, AlexConfig, AlexVisionConfig, AlexVisionProjectionConfig
from .processing_alex import AlexProcessor


def load_preprocessor(config: dict):
    """
    Load preprocessor from the config for training the model.
    """
    pretrain_name = config['pretrain_name']
    # load preprocessor first
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    # TODO: These values only work for CLIP. Make it more general
    vision_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    processor = AlexProcessor(
        tokenizer=tokenizer, 
        image_processor=vision_processor,
        **config['preprocessor']
    )
    return processor


def load_model_and_preprocessor(config: Dict[str, Any]):
    """
    Load model and preprocessor from the config for training the model
    based on the HuggingFace weights.

    When loading the weights you trained, don't use this function.
    Instead just use AlexOPTForAction.from_pretrained(your_model_path).

    Contents of the config:
        - pretrain_name: The name of the pre-trained model for the langauge model.
        - vision_pretrain_name: The name of the pre-trained model for the vision model.
        - preprocessor: The configuration for the preprocessor.
        - model: The configuration for the model.
    """
    # load preprocessor
    processor = load_preprocessor(config)

    # name for the pre-trained models
    pretrain_name = config['pretrain_name']                 # language model
    vision_pretrain_name = config['vision_pretrain_name']   # vision model

    # token info for the model
    config['model']['frame_emb_token_id'] = processor.frame_emb_token_id
    config['model']['frame_end_token_id'] = processor.frame_end_token_id
    
    # prepare config for the model
    projection_config = AlexVisionProjectionConfig(**config['vision_projection'])
    vision_config = AlexVisionConfig.from_dict_and_pretrained(
        config['vision'], pretrained_name=vision_pretrain_name
    )
    alex_config = AlexConfig.from_dict_and_pretrained(
        config['model'],
        vision_config=vision_config,
        vision_projection_config=projection_config,
        pretrained_name=pretrain_name
    )

    # instantiate the model and load weights for the language model
    model = AlexOPTForAction.from_pretrained(
        pretrained_model_name_or_path=pretrain_name, 
        config=alex_config
    )
    # load weights for the vision model
    # This is double initialization but it is necessary to load the weights
    model.vision_model.model = CLIPVisionModel.from_pretrained(vision_pretrain_name)

    # Expand the embedding layer for the language model
    model.resize_token_embeddings(len(processor.tokenizer))
    return model, processor