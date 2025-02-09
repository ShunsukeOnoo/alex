"""
Implementation of the Alex model based on OPT model.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from einops import rearrange
from transformers import CLIPVisionModel, CLIPVisionConfig, PreTrainedModel, PretrainedConfig
from transformers.utils import logging, ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTDecoder, OPTModel, OPTForCausalLM

logger = logging.get_logger(__name__)


class AlexVisionConfig(CLIPVisionConfig):
    """
    Configuration class for the vision encoder module.

    Suppose using the pretrained configuration of CLIPVisionModel by using from_pretrained() method.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AlexVisionProjectionConfig(PretrainedConfig):
    """
    Configuration class for the vision projection module.

    Args:
        projection_type (str): Type of the projection. Default is 'linear'.
        input_dim (int): Dimension of the input embeddings of the projection module.
            The input tensor will have the shape (batch_size, n_frames, tokens_per_frame, input_dim).
        emb_dim (int): Dimension of the output embeddings. The output tensor will have the shape
            (batch_size, n_frames, tokens_per_frame, emb_dim).
    """
    def __init__(self, projection_type: str, input_dim: int, emb_dim: int):
        self.projection_type = projection_type
        self.input_dim = input_dim
        self.emb_dim = emb_dim


class AlexConfig(OPTConfig):
    """
    Configuration class to store the configuration of the OPT based Alex model.

    Args (unique to AlexConfig):

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_config(
            self, 
            vision_config: AlexVisionConfig,
            vision_projection_config: AlexVisionProjectionConfig,
            frame_emb_token_id: int,
            frame_end_token_id: int,
            action_dim: int = 22,  # TODO: Check this value
            binary_action_dims: List[int] = None,
            analogue_action_dims: List[int] = None,
            timestamp_embedding: str = 'time2vec'
            ) -> None:
        """
        Add configuration information for the Alex model.

        Args:
        vision_config (AlexVisionConfig): Configuration for the vision encoder module.
        vision_projection_config (AlexVisionProjectionConfig): Configuration for the vision projection module.
        frame_emb_token_id (int): Token ID for the video frame embedding.
        frame_end_token_id (int): Token ID for the end of the video frame embedding.
        action_dim (int): Dimension of the action prediction.
        binary_action_dims (List[int]): List of indices of the binary action dimensions.
        analogue_action_dims (List[int]): List of indices of the analogue action dimensions.
        timestamp_embedding (str): Type of the timestamp embedding. Default is 'time2vec'.  
        """
        self.vision_config = vision_config
        self.vision_projection_config = vision_projection_config
        self.frame_emb_token_id = frame_emb_token_id
        self.frame_end_token_id = frame_end_token_id
        self.action_dim = action_dim
        self.binary_action_dims = binary_action_dims
        self.analogue_action_dims = analogue_action_dims
        self.timestamp_embedding = timestamp_embedding


@dataclass
class AlexModelOutput(ModelOutput):
    """
    Output of the Alex model.
    In addition to the original CausalLMOutputWithPast, it includes:
    - action_loss: Loss of the action prediction.
    - lm_loss
    - action_logits: Logits of the action prediction.
    """
    # from CausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # Additional outputs
    action_loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    action_logits: Optional[torch.Tensor] = None


class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super(Time2Vec, self).__init__()
        self.d_model = d_model
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.zeros(d_model - 1))
        self.B = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t):
        t = t.unsqueeze(-1)  # Shape: (batch_size, n_tokens, 1)
        v1 = self.w0 * t + self.b0  # Shape: (batch_size, n_tokens, 1)
        v2 = torch.sin(t * self.W + self.B)  # Shape: (batch_size, n_tokens, d_model-1)
        return torch.cat([v1, v2], -1)  # Shape: (batch_size, n_tokens, d_model)
    

class AlexVisionEncoder(PreTrainedModel):
    """
    Embed video frames using vision model.
    
    Contents of the config:
        (config for the model to be used)
        use_last_projection (bool): Whether to use the last projection output of the model
    """
    def __init__(self, config: AlexVisionConfig):
        super().__init__(config)
        self.config = config
        # Instantiate the vision model without loading weights
        self.model = CLIPVisionModel(config)

    def forward(self, video_frames: torch.Tensor):
        """
        Args:
            video_frames: Shape (batch_size, n_frames, channel, height, width). 
                Preprocessed video frames. May contain padding frames whose values are zeros.
        Returns:
            torch.Tensor: Shape (batch_size, n_frames, tokens_per_frame, hidden_size).
        """
        # Turn the input shape into (batch_size * n_frames, channel, height, width)
        batch_size, n_frames, channel, height, width = video_frames.size()
        video_frames = video_frames.flatten(0, 1)

        # Use the model to produce the embeddings
        out = self.model(pixel_values=video_frames, return_dict=True)

        # Based on the config, return what is needed.
        if self.config.use_last_projection:
            vision_embeds = out.pooler_output  # (batch_size * n_frames, hidden_size)
            return vision_embeds.view(batch_size, n_frames, -1).unsqueeze(2)  # (batch_size, n_frames, 1, hidden_size)
        else:
            vision_embeds = out.last_hidden_state # (batch_size * n_frames, *, hidden_size)
            _, n_embeds, hidden_size = vision_embeds.size()
            return vision_embeds.view(batch_size, n_frames, n_embeds, hidden_size)  # (batch_size, n_frames, *, hidden_size)


class AlexVisionProjection(PreTrainedModel):
    """
    Turn the output of the vision encoder into the input of the language model.
    TODO: Implement more sophisticated projection methods in the future.

    Config:
        projection_type (str): Type of the projection. Default is 'linear'.
        input_dim (int): Dimension of the input embeddings.
        emb_dim (int): Dimension of the output embeddings.
    """
    def __init__(self, config: AlexVisionProjectionConfig):
        super().__init__(config)
        self.config = config

        if config.projection_type.lower() == 'linear':
            self.projection = nn.Linear(config.input_dim, config.emb_dim)

        elif config.projection_type.lower() == 'resampler':
            raise NotImplementedError("Resampler is not implemented yet.")
        
        else:
            raise ValueError(f"Unknown projection type: {config.projection_type}")

    def forward(self, video_embeds):
        """
        Args:
            video_embeds: Shape (batch_size, n_frames, tokens_per_frame, input_dim)
        Returns:
            torch.Tensor: Shape (batch_size, n_frames, tokens_per_frame, emb_dim)
        """
        return self.projection(video_embeds)
    

class AlexOPTDecoder(OPTDecoder):
    """
    Base decoder transformer for the Alex model.
    Wraps the OPTDecoder and add unique timestamp embeddings.
    """
    def __init__(self, config: AlexConfig):
        super().__init__(config)

        if config.timestamp_embedding.lower() == 'time2vec':
            self.embed_timestamp = Time2Vec(config.hidden_size)
        else:
            raise ValueError(f"Unknown timestamp embedding: {config.timestamp_embedding}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timestamps: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        time_embeds = self.embed_timestamp(timestamps)
        # pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + time_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class AlexOPTModel(OPTModel, OPTPreTrainedModel):
    def __init__(self, config: AlexConfig):
        super(OPTPreTrainedModel, self).__init__(config)
        self.decoder = AlexOPTDecoder(config)
        self.post_init()

    def forward(
            self,
            timestamps,
            attention_mask,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            past_key_values,
            return_dict,
        ):
        decoder_outputs = self.decoder(
            timestamps=timestamps,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            return_dict=return_dict
        )
        return decoder_outputs


class AlexOPTForAction(OPTForCausalLM, OPTPreTrainedModel):
    """
    Alex model that predicts actions on the input video + text sequence.

    Args:
        config (AlexConfig): Configuration class for the model, which includes the following parameters.
            vocab_size: int Number of tokens in the vocabulary including the frame tokens.
            vision_config
            vision_projection_config
            action_dim
            binary_action_dims: List of int. Dimension of the analogue actions.
            analogue_action_dims: List of int. Dimension of the binary actions.
            timestamp_embedding: str. Type of the timestamp embedding.
            frame_emb_token_id (int): Token ID for the video frame embedding.
            frame_end_token_id (int): Token ID for the end of the video frame embedding.
    """
    def __init__(self, config: AlexConfig):
        super(OPTPreTrainedModel, self).__init__(config)
        # Original modules for the OPT model
        self.model = AlexOPTModel(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Additional modules for the Alex model
        self.vision_model = AlexVisionEncoder(config.vision_config)
        self.vision_projection = AlexVisionProjection(config.vision_projection_config)
        self.action_head = nn.Linear(config.hidden_size, config.action_dim, bias=False)
        
        self.binary_action_dims = config.binary_action_dims
        self.analogue_action_dims = config.analogue_action_dims
        self.binary_action_loss = nn.BCEWithLogitsLoss()
        self.analogue_action_loss = nn.MSELoss()

    def forward(
            self,
            input_ids: torch.Tensor = None,
            video_frames: torch.Tensor = None,
            timestamps: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            actions: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple, AlexModelOutput]:
        """Forward pass of the model.
    
        Note that n_tokens and n_frames are different. n_token is the number of text tokens and video frames in the sequence.

        Args:
            input_ids: Shape (batch_size, n_tokens). Input tokens which include video placeholders.
            video_frames: Shape (batch_size, n_frames, height, width). Preprocessed video frames. May contain padding frames 
                whose values are zeros.
            timestamps: Shape (batch_size, n_tokens). Timestamps of the tokens.
            attention_mask: Shape (batch_size, n_tokens). 2D attention mask which indicates the position of the padding tokens.
            past_key_values: List of torch.Tensor. Used for fast decoding.
            actions: Shape (batch_size, n_tokens, action_dim). Target actions.
            labels: Shape (batch_size, n_tokens). Target labels for text.
            use_cache: bool. Used for fast decoding.
            output_attentions: bool. Whether to output attentions.
            output_hidden_states: bool. Whether to output hidden states.
            return_dict: bool. Whether to return a dictionary.
        """
        # Embed video frames
        video_embeds = self.vision_model(video_frames)
        video_embeds = self.vision_projection(video_embeds)
        # (batch_size, n_frames, tokens_per_frame, emb_dim): 

        # Embed text tokens
        text_embeds = self.model.decoder.embed_tokens(input_ids)

        # Combine the vision and text embeddings
        frame_emb_mask = input_ids == self.config.frame_emb_token_id
        inputs_embeds = combine_embeddings(text_embeds, video_embeds, frame_emb_mask)
        # (batch_size, n_tokens, emb_dim)

        # Call the base Transformer model
        outputs = self.model.forward(
            timestamps=timestamps,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            return_dict=return_dict
        )
        # outputs[0]: (batch_size, n_tokens, hidden_size)
        
        # Predict next tokens
        lm_logits = self.lm_head(outputs[0]).contiguous()        # (batch_size, n_tokens, vocab_size)

        # Predict actions
        action_logits = self.action_head(outputs[0]).contiguous() # (batch_size, n_tokens, action_dim)

        # Calculate loss
        loss = None
        if actions is not None:
            # check the location of the action predictions: they are end of each video frame
            action_target_mask = input_ids == self.config.frame_end_token_id
            action_loss = calculate_action_loss(
                action_logits, actions, action_target_mask, self.binary_action_dims, self.analogue_action_dims,
                self.binary_action_loss, self.analogue_action_loss)
            loss = action_loss
        else:
            action_loss = None
        if labels is not None:
            # check the location of the text token predictions
            # they are tokens whose next token is not a video frame token
            text_target_mask = attention_mask * (input_ids != self.config.frame_emb_token_id) * (input_ids != self.config.frame_end_token_id)
            text_target_mask = torch.roll(text_target_mask, 1, dims=-1)
            text_target_mask[:, -1] = False
            lm_loss = calculate_lm_loss(lm_logits, labels, text_target_mask)
            loss = lm_loss if loss is None else loss + lm_loss
        else:
            lm_loss = None

        # return the output
        if not return_dict:
            return (loss, action_loss, lm_loss, action_logits, lm_logits) + outputs[1:]
        return AlexModelOutput(
            loss=loss,
            action_loss = action_loss,
            lm_loss = lm_loss,
            action_logits=action_logits,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
            

def combine_embeddings(
        text_embeds: torch.Tensor,
        video_embeds: torch.Tensor,
        frame_emb_mask: torch.Tensor,
):
    """
    Insert video embeddings into the text embeddings.
    Note:
        Each sequence may has different number of video frames.

    Args:
        text_embeddigs: Shape (batch_size, n_tokens, emb_dim)
        video_embeddings: Shape (batch_size, n_frames, token_per_frame, emb_dim)
        frame_emb_mask: Bool tensor with shape (batch_size, n_tokens). Indicates the position of the frame embeddings in the sequence.

    Returns:
        torch.Tensor: Shape (batch_size, n_tokens, emb_dim)
    """
    # TODO: Find more efficient implementation
    batch_size = text_embeds.size(0)
    n_frame_tokens = frame_emb_mask.sum(dim=1)  # (batch_size)
    for i in range(batch_size):
        text_embeds[i, frame_emb_mask[i]] = video_embeds[i, :n_frame_tokens[i]].flatten(0, 1)
    return text_embeds


def calculate_action_loss(action_logits, action_targets, action_target_mask, binary_action_dims, analogue_action_dims,
                        binary_loss, analogue_loss):
    """
    Calculate the action loss.

    Args:
        action_logits: Shape (batch_size, n_tokens, action_dim)
        action_targets: Shape (batch_size, n_frames, action_dim)
        action_target_mask: Shape (batch_size, n_tokens)
    """
    # Calculate the loss for each sample
    # because the number of frames is different for each sample.
    # TODO: More efficient implementation
    loss = 0
    batch_size, n_tokens, action_dim = action_logits.size()

    # the number of actions for each sample is different
    # that is why we need to calculate the loss for each sample
    # TODO: can we calculate the loss for all samples at once and 
    # then mask out and take the mean?

    for i in range(batch_size):
        n_frames = action_target_mask[i].sum().item()
        loss += binary_loss(
            action_logits[i, action_target_mask[i]][:, binary_action_dims],   # (n_frames, n_binary_actions)
            action_targets[i, :n_frames, binary_action_dims]                # (n_frames, n_binary_actions)
        )
        loss += analogue_loss(
            action_logits[i, action_target_mask[i]][:, analogue_action_dims],  # (n_frames, n_analogue_actions)
            action_targets[i, :n_frames, analogue_action_dims]
        )
    return loss


def calculate_lm_loss(lm_logits, labels, text_target_mask):
    """
    Calculate the language model loss.

    Args:
        lm_logits: Shape (batch_size, seq_len?, vocab_size)
        labels: Shape (batch_size, seq_len?)
        text_target_mask: Shape (batch_size, seq_len?)
    """
    # flatten batch dimension
    lm_logits = rearrange(lm_logits, 'b s v -> (b s) v')
    labels = rearrange(labels, 'b s -> (b s)')
    text_target_mask = rearrange(text_target_mask, 'b s -> (b s)')

    # retrieve the target tokens
    lm_logits = lm_logits[text_target_mask]
    labels = labels[text_target_mask]

    loss_func = nn.CrossEntropyLoss()
    return loss_func(lm_logits, labels)