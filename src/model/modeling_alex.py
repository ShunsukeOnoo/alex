"""
Implementation of the Alex model.
Somewhat I refer to the implementation of the Kosmos-2 model on HuggingFace.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn.modules.module import Module
import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class AlexConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of the Alex model.

    Subconfigs
        text_config
        vision_config
        vision_projection_config

    TODO: Implement this
    """
    pass

class AlexTextConfig(PretrainedConfig):
    pass

class AlexVisionConfig(PretrainedConfig):
    pass

class AlexVisionProjectionConfig(PretrainedConfig):
    pass


class AlexModelOutput(ModelOutput):
    """
    What I need to include:
        return AlexModelOutput(
            loss=action_loss + lm_loss,
            action_logits=action_logits,
            lm_logits=lm_output.logits,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions
        )
    """
    loss: Optional[torch.Tensor] = None 
    action_loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    action_logits: Optional[torch.Tensor] = None
    lm_logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None


class AlexPreTrainedModel(PreTrainedModel):
    """
    This is an abstract class that handles the loading of pretrained weights.
    TODO: Implement this.
    """
    pass


class AlexTextModel(AlexPreTrainedModel):
    """
    Base Alex model that handles the merging of the text and vision embeddings.
    """
    def __init__(self, config):
        self.config = config
        self.transformer = ...

    def forward(
            self,
            input_ids: torch.Tensor = None,
            timestamps: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            video_embeds: torch.Tensor = None,
            video_mask: torch.Tensor = None,
            text_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ):
        # Embed text
        if text_embeds is None:
            text_embeds = self.embed_tokens(input_ids)

        # Merge video embeddings with text embeddings
        input_embeds = insert_video_embeddings(text_embeds, video_embeds, video_mask)

        # Add positional embeddings
        pos_embeds = embed_temporal_positions(timestamps)
        input_embeds = input_embeds + pos_embeds

        # Input embeddings to the transformer
        return self.transformer(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
    

class AlexVisionEncoder(AlexPreTrainedModel):
    """
    Embed video frames using a pretrained vision model.
    """
    def __init__(self, config: AlexVisionConfig):
        super().__init__(config)
        self.vision_model = ...
    def forward(self, video_frames: torch.Tensor):
        """
        Args:
            video_frames: Shape (batch_size, n_frames, height, width)
        Returns:
            torch.Tensor: Shape (batch_size, n_frames, n_seq, emb_dim)
        """
        # TODO: Implement this.
        pass 


class AlexVisionProjection(AlexPreTrainedModel):
    def __init__(self, config: AlexVisionProjectionConfig):
        super().__init__(config)
        self.projection = ...
    def forward(self, video_embeds):
        """
        Args:
            video_embeds: Shape (batch_size, n_frames, n_seq, emb_dim)
        Returns:
            torch.Tensor: Shape (batch_size, n_frames, emb_dim)
        """
        return self.projection(video_embeds)


class AlexForActionGeneration(AlexPreTrainedModel):
    """
    Alex model that predicts actions on the input video + text (optional).
    """

    def __init__(self, config: AlexConfig):
        super().__init__(config)

        self.lang_model = AlexTextModel(config.text_config)
        self.vision_model = AlexVisionEncoder(config.vision_config)
        self.image_projection = AlexVisionProjection(config.vision_projection_config)
        self.action_head = nn.Linear(config.hidden_size, config.action_dim)
        # TODO: We may want to apply a non-linear activation function here.

        # TODO: Expand input embeddings to include video placeholders.

    def forward(
            self,
            input_ids: torch.Tensor = None,
            video_frames: torch.Tensor = None,
            timestamps: torch.Tensor = None,
            actions: Optional[torch.Tensor] = None,
            action_target_mask: torch.Tensor = None,
            text_target_mask: torch.Tensor = None,
            video_frame_mask: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple, AlexModelOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids: Shape (batch_size, seq_len)
            video_frames: Shape (batch_size, n_frames, height, width)
            timestamps: Shape (batch_size, seq_len)
            actions: Shape (batch_size, seq_len, action_dim). Target actions.
            action_target_mask: Shape (batch_size, seq_len). Indicates the position of the action target.
            text_target_mask: Shape (batch_size, seq_len). Indicates the position of the text target.
            video_frame_mask: Shape (batch_size, n_frames). Indicates the position of the video frames in the sequence.
                0 for non-video tokens, 1 for video tokens.
            attention_mask: Shape (batch_size, seq_len). Indicates the position of the padding tokens.
            past_key_values: List of torch.Tensor. Used for fast decoding.
            labels: Shape (batch_size, seq_len). Target labels for text.
            use_cache: bool. Used for fast decoding.
            output_attentions: bool. Whether to output attentions.
            output_hidden_states: bool. Whether to output hidden states.
            return_dict: bool. Whether to return a dictionary.
        """
        # Embed vision
        video_embeds = self.vision_model(video_frames)
        video_embeds = self.image_projection(video_embeds)
        # Shape (batch_size, n_frames, emb_dim)
        # TODO: Match the embedding size with the hidden size of the language model.

        # call the base model
        lm_output = self.text_model(
            input_ids=input_ids,
            timestamps=timestamps,
            attention_mask=attention_mask,
            video_frame_mask=video_frame_mask,
            video_embeds=video_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Calculate the action logits
        action_logits = self.action_head(lm_output.last_hidden_state)
        # Retrieve the action logits for the target positions
        # TODO: Review the dtype of action_target_mask
        # TODO: Number of actions may vary for each sample.
        action_logits = action_logits[action_target_mask]
        # Shape (batch_size, seq_len?, action_dim)

        # calculate the action loss
        action_loss = ...

        # calculate the language model loss
        lm_loss = ...

        # return the output
        return AlexModelOutput(
            loss=action_loss + lm_loss,
            action_loss = action_loss,
            lm_loss = lm_loss,
            action_logits=action_logits,
            lm_logits=lm_output.logits,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions
        )


def embed_temporal_positions(timestamps: torch.Tensor):
    """
    Embed temporal positions using continuous timestamps.

    Args:
        timestamps: Shape (batch_size, seq_len), elements shows the timestamps of the tokens.
    Returns:
        torch.Tensor: Shape (batch_size, seq_len, emb_dim)
    """
    # TODO: Implement this.
    pass


def insert_video_embeddings(
        text_embeds: torch.Tensor,
        video_embeds: torch.Tensor,
        video_mask: torch.Tensor,
):
    """
    Insert video embeddings into the text embeddings.
    Note:
        Each sequence may has different number of video frames.

    Args:
        text_embeddigs: Shape (batch_size, seq_len, emb_dim)
        video_embeddings: Shape (batch_size, n_frames, emb_dim)
        video_frame_mask: bool tensor with shape (batch_size, n_frames)

    Returns:
        torch.Tensor: Shape (batch_size, seq_len, emb_dim)
    """
    # TODO: Come up with a method without a loop.
    # Iterate over the batch: not efficient
    embeddings = torch.clone(text_embeds)
    for i in range(embeddings.size(0)):
        n_frames = video_mask[i].sum().item()
        embeddings[i, video_mask[i]] = video_embeds[i, :n_frames]
    
    return embeddings