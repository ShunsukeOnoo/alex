"""
Implementation of the Alex model based on OPT model.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from einops import rearrange
from transformers import PretrainedConfig
from transformers.utils import logging, ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTDecoder, OPTModel, OPTForCausalLM


logger = logging.get_logger(__name__)


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
    

class AlexOPTDecoder(OPTDecoder):
    def __init__(self, config: AlexConfig):
        super().__init__(config)

        if config.timestamp_embedding.lower() == 'time2vec':
            # TODO: Check the dimension of the timestamp embedding.
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


class AlexOPTForAction(OPTForCausalLM, OPTPreTrainedModel):
    """
    Alex model that predicts actions on the input video + text (optional).
    """

    def __init__(self, config: AlexConfig):
        super(OPTPreTrainedModel, self).__init__(config)
        # Original modules for the OPT model
        self.model = AlexOPTModel(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Additional modules for the Alex model
        self.vision_model = AlexVisionEncoder(config.vision_config)
        self.image_projection = AlexVisionProjection(config.vision_projection_config)
        self.action_head = nn.Linear(config.hidden_size, config.action_dim, bias=False)
        # TODO: We may want to apply a non-linear activation function after action_head.
        # TODO: Expand input embeddings of the language model to include video placeholders.
        # TODO: the output dim of the vision_projection should be the same as the hidden_size of the language model.

    def forward(
            self,
            input_ids: torch.Tensor = None,
            video_frames: torch.Tensor = None,
            timestamps: torch.Tensor = None,
            actions: Optional[torch.Tensor] = None,
            action_target_mask: torch.Tensor = None,
            text_target_mask: torch.Tensor = None,
            video_frame_mask: torch.Tensor = None,
            video_emb_mask: torch.Tensor = None,
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
            # TODO: Each sample may contains different number of video frames...
            # TODO: The tensor may contains padding values at the end.
            video_frames: Shape (batch_size, n_frames, height, width)
            timestamps: Shape (batch_size, seq_len)
            actions: Shape (batch_size, seq_len, action_dim). Target actions.
            action_target_mask: Shape (batch_size, seq_len). Indicates the position of the action target.
            text_target_mask: Shape (batch_size, seq_len). Indicates the position of the text target.
            # TODO: The video embeddings may span multiple tokens.
            video_frame_mask: Shape (batch_size, seq_len). Indicates the position of the video frames in the sequence.
                0 for non-video tokens, 1 for video tokens.
            video_emb_mask: Shape (batch_size, n_frames). Indicates the position of the video embeddings 
                in video_frames.
            attention_mask: Shape (batch_size, seq_len). Indicates the position of the padding tokens.
            past_key_values: List of torch.Tensor. Used for fast decoding.
            labels: Shape (batch_size, seq_len). Target labels for text.
            use_cache: bool. Used for fast decoding.
            output_attentions: bool. Whether to output attentions.
            output_hidden_states: bool. Whether to output hidden states.
            return_dict: bool. Whether to return a dictionary.
        """
        # Embed inputs
        video_embeds = self.vision_model(video_frames)
        video_embeds = self.image_projection(video_embeds)
        # TODO: Check the embedding size of the language model.
        # Shape (batch_size, n_frames, emb_dim), where emb_dim corresponds to
        # the embedding size of the language model.
        text_embeds = self.model.decoder.embed_tokens(input_ids)

        # Combine the vision and text embeddings
        input_embeds = insert_video_embeddings(text_embeds, video_embeds, video_frame_mask, video_emb_mask)

        # Call the base Transformer model
        # TODO: Check the parameters
        outputs = self.model.decoder(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            timestamps=timestamps,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # Apply lm_head and action_head
        lm_logits = self.lm_head(outputs[0]).contiguous()        # (batch_size, seq_len, vocab_size)
        action_logits = self.action_head(outputs[0]).contiguous() # (batch_size, seq_len, action_dim)

        # calculate losses
        oveall_loss = None
        if actions is not None:
            # TODO: Reconsider the dtype of action_target_mask
            action_loss = calculate_action_loss(action_logits, actions, action_target_mask)
            overall_loss = action_loss
        else:
            action_loss = None
        if labels is not None:
            lm_loss = calculate_lm_loss(lm_logits, labels, text_target_mask)
            overall_loss = lm_loss if overall_loss is None else overall_loss + lm_loss
        else:
            lm_loss = None

        # return the output
        if return_dict:
            return AlexModelOutput(
                loss=overall_loss,
                action_loss = action_loss,
                lm_loss = lm_loss,
                action_logits=action_logits,
                lm_logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        else:
            # TODO: Implement this.
            pass


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
        video_frame_mask: torch.Tensor,
        video_emb_mask: torch.Tensor = None
):
    """
    Insert video embeddings into the text embeddings.
    Note:
        Each sequence may has different number of video frames.

    Args:
        text_embeddigs: Shape (batch_size, seq_len, emb_dim)
        video_embeddings: Shape (batch_size, n_frames, emb_dim)
        video_frame_mask: bool tensor with shape (batch_size, seq_len)
        video_emb_mask: bool tensor with shape (batch_size, n_frames)

    Returns:
        torch.Tensor: Shape (batch_size, seq_len, emb_dim)
    """
    # TODO: consider better names for video_frame_mask, because it's rather a mask for input_ids.
    if not video_emb_mask:
        b, f, e = video_embeds.size()
        video_emb_mask = torch.zeros((b, f), dtype=torch.bool, device=video_embeds.device)
        for i in range(b):
            n_frames = video_frame_mask[i].sum().item()
            video_emb_mask[i, :n_frames] = True
        # video_emb_mask: (batch_size, n_frames)

    text_embeds[video_frame_mask] = video_embeds[video_emb_mask]
    return text_embeds


def calculate_action_loss(action_logits, action_targets, action_target_mask):
    """
    Calculate the action loss.

    Args:
        action_logits: Shape (batch_size, seq_len, action_dim)
        action_targets: Shape (batch_size, seq_len, action_dim)
        action_target_mask: Shape (batch_size, seq_len)
    """
    # flatten batch dimension
    action_logits = rearrange(action_logits, 'b s a -> (b s) a')
    action_targets = rearrange(action_targets, 'b s a -> (b s) a')
    action_target_mask = rearrange(action_target_mask, 'b s -> (b s)')

    # retrieve the target actions
    action_logits = action_logits[action_target_mask]
    action_targets = action_targets[action_target_mask]
    
    # TODO: Use sophisticated loss function. For some actions, we may want to use MSE loss.
    loss_func = nn.CrossEntropyLoss()
    return loss_func(action_logits, action_targets)


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