"""
Implementation of the Alex model.
Somewhat I refer to the implementation of the Kosmos-2 model on HuggingFace.
"""
from typing import Callable
import torch
from torch import nn
from torch.nn.modules.module import Module
import transformers


class AlexConfig(transformers.PretrainedConfig):
    """
    What I need to include:
    - use_return_dict
    """
    pass


class AlexModelOutput(transformers.ModelOutput):
    """
    What I need to include:
    - loss: (lm_loss, action_loss)?
    - logits: (lm_logits, action_logits)?
    - past_key_values
    - hidden_states
    - attentions
    """
    pass


class AlexPreTrainedModel(transformers.PreTrainedModel):
    """
    This is an abstract class that handles the loading of pretrained weights.
    """
    pass


class AlexModel(AlexPreTrainedModel):
    """
    Alex model that plays Minecraft.

    To do:
    - implement the alignment of the video embeddings and the text embeddings
    - implement the loss calculation
    - implement the video embedding
    - implement the generation
    """

    def __init__(self, config):
        self.config = config

        # submodules
        # token_embeddings must be consistent with the transformer
        self.vision_encoder = ...
        self.embed_tokens = ...   # text embeddings
        self.embed_positions = ...    # positional embeddings
        self.transformer = ...        # causal transformer language model without head
        self.lm_head = ...
        self.action_head = ...


    def forward(
            self,
            input_ids=None,
            input_token_embeds=None,
            video_frames=None,
            token_timestamps=None,
            video_indices=None,
            video_embeds=None,
            past_key_values=None,
            lm_labels=None,
            action_labels=None,
            output_attentions=None,
            return_dict=None            
            ):
        """
        Combine the video frames and the text tokens to generate the next action.

        Args:
            input_ids (torch.Tensor): 
                The input text tokens.
            input_token_embeds (torch.Tensor):
                The input text embeddings.
            video_frames (torch.Tensor):
                The input video frames.
            video_embeds (torch.Tensor):
                The input video embeddings.
            past_key_values (tuple):
                The past key values of the transformer.
            lm_labels (torch.Tensor):
                The language modeling labels.
            action_labels (torch.Tensor):
                The action labels.
            output_attentions (bool):
                Whether to output attentions.
            return_dict (bool):
                Whether to return a dictionary.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # embed video frames with vision model
        if video_embeds is None:
            video_embeds = self.embed_video_frames(video_frames)

        # embed text tokens if provided
        if input_token_embeds is None:
            input_token_embeds = self.embed_tokens(input_ids)

        # align the video embeddings with the text embeddings
        input_embeds = self.align_embeddings(input_token_embeds, video_embeds)

        # add positiional (temporal) embeddings to the both embeddings
        input_embeds = input_embeds + self.embed_positions(input_embeds)

        # pass the embeddings to the language model
        outputs = self.transformer(
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            return_dict=return_dict
        )

        # apply the language head and action head
        lm_logits = self.lm_head(outputs[0])
        action_logits = self.action_head(outputs[0])

        # compute the action loss
        lm_loss = self.calculate_lm_loss(lm_logits, lm_labels)
        action_loss = self.calculate_action_loss(action_logits, action_labels)

        if not return_dict:
            output = (lm_logits, action_logits) + outputs[1:]
            return ((lm_loss, action_loss) + output) if lm_loss is not None else output
        
        return AlexModelOutput(
            loss=(lm_loss, action_loss),
            logits=(lm_logits, action_logits),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def embed_video_frames(self, video_frames):
        """
        Embed the video frames with the vision encoder.
        Note that the video frames are already preprocessed.

        Args:
            video_frames (torch.Tensor):
                The video frames.
        Returns:
            video_embeds (torch.Tensor):
                The video embeddings.
        """
        pass

    def align_embeddings(self, input_embeds, video_embeds):
        """
        Align the video embeddings with the text embeddings and
        return the interleaved embeddings.

        Args:
            input_embeds (torch.Tensor):
                The text embeddings.
            video_embeds (torch.Tensor):
                The video embeddings.
        Returns:
            embeds (torch.Tensor):
                The interleaved embeddings.
        """
        pass

    def generate(self):
        """
        Generate text from interleaved vision + text
        """
        # we need to add GenerationMixIn to do this
        pass

    def play(self, env, instruction=None):
        """
        Play the game by iteratively running the model.

        Args:
            env (): The MineDojo environment.
            instruction (str): The instruction to be executed.
        """
        # 
        is_done = False
        past_key_values = None

        # if there is an instruction, we first run the model on instruction
        # first, we need to tokenize the instruction 
        # and add the special token <play> at the beginning (or end?)
        out = ...
        past_key_values = ...

        while not is_done:
            # get the current state (video frame) of the environment

            # preprocess the video frame

            # run the model on the video frame to get the next action
            # (don't forget to pass the past_key_values)
            action = ...

            # take the action in the environment

            # check if the game is done
            is_done = ...

        # do some postprocessing
        pass