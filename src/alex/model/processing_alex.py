"""
Preprocessing module for the Alex model.

TODO
- Consider making a class for action.

"""

from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
# from transformers.processing_utils import ProcessorMixin


class AlexProcessor:
    """Processor for the Alex model.

    There are three special tokens:
    - frame_start_token: Shows the start of the video frames.
    - frame_emb_token: Placeholder for the video embeddings.
    - frame_end_token: Shows the end of the video frames.
    The variable '..._token' indicates the token string, and '..._token_id' indicates the tokenized id (int).

    frame_tokens: The string that represents a single frame. It is a concatenation of the three special tokens.
    - frame_tokens = frame_start_token + frame_emb_token * frame_emb_len + frame_end_token
    - frame_token_ids: Tokenized ids for the frame_tokens.
    - frame_token_len: Length of the frame_tokens.


    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for the language model.
        image_processor (callable, optional): Image processor for the video frames.

        frame_start_token (str): Start token for the video frames.
        frame_emb_token (str): Placeholder for the video embeddings.
        frame_end_token (str): End token for the video frames.
        
        frame_emb_len (int): Length of the frame embeddings.
            The overall placeholder for a single frame will be frame_token * (frame_emb_len - 1) + frame_end_token.
        default_fps (int): Default frame per second for the video clips. Used when frame_timestamps is not provided.
        pad_value (float): Value to pad the sequences other than input_ids.
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 image_processor: Optional[Callable] = None,
                 frame_start_token: str = "<frame/>",
                 frame_emb_token: str = "<frame>", 
                 frame_end_token: str = "</frame>",
                 frame_emb_len: int = 1,
                 default_fps: int = 30,
                 return_labels: bool = True,
                 ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.frame_start_token = frame_start_token
        self.frame_token = frame_emb_token
        self.frame_end_token = frame_end_token
        self.frame_emb_len = frame_emb_len
        self.default_fps = default_fps
        self.return_labels = return_labels

        assert frame_emb_token != '', "frame_token must not be an empty string."
        assert frame_end_token != '', "frame_end_token must not be an empty string."
        assert frame_emb_len > 0, "frame_emb_len must be a positive integer."

        # Add special tokens
        self.num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': [frame_start_token, frame_emb_token, frame_end_token]})
        
        # token for a single embedding
        self.frame_emb_token_id = tokenizer.convert_tokens_to_ids(frame_emb_token)
        
        # last token for a single frame
        self.frame_end_token_id = tokenizer.convert_tokens_to_ids(frame_end_token)

        # all tokens that represent a single frame
        self.frame_tokens = frame_start_token + frame_emb_token * frame_emb_len + frame_end_token
        self.frame_token_ids = tokenizer.encode(self.frame_tokens, add_special_tokens=False)  # List[int]
        self.frame_token_len = len(self.frame_token_ids)  # length of total special tokens per frame TODO: Name is confusing.
    
    def __call__(
            self,
            video_frames: torch.Tensor,
            frame_timestamps,
            text: list[str],
            text_timestamps: list[tuple[float, float]],
            actions: list[dict] = None,
            return_labels: bool = None,
            unsqueeze: bool = False
        ):
        """
        Process and input sample and return input to the model.
        Can be used as transform for the YouTubeDataset. 

        Args:
            video_frames (torch.Tensor): Video frames for the video clip. (n_frame, num_channel, height, width) with range [0, 1].
            frame_timestamps (Union[list[float], torch.tensor]): Timestamps for video frames in seconds.
            text (list[str]): Transcripts of the video clip.
            text_timestamps (list[tuple[float, float]]): The start time and end time of each transcript.
            actions (list[dict]): (optional) Training targets. List of actions that corresponds to each video frame.
            return_labels (bool): Whether or not to return the labels for input_ids. Overrides the default value.\
            unsqueeze (bool): Whether to unsqueeze the tensors to make them batched.

        Returns:
            dict: Dictionary that contains the following keys:
                video_frames (torch.Tensor): Preprocessed video frames with shape (1, n_frames, num_channel, height, width).
                timestamps (torch.Tensor): Timestamps for each tokens. Shape (1, n_tokens).
                input_ids (torch.Tensor): Token ids that contians the video frame placeholders. Shape (1, n_tokens).
                actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to each video frames.
                    (1, n_frames, action_dim).
                labels (torch.Tensor): (Optional) Labels for input_ids. Shape (1, n_tokens-1)
        """
        # process the sample
        inputs = self.process_sample(
            video_frames=video_frames,
            frame_timestamps=frame_timestamps,
            text=text,
            text_timestamps=text_timestamps,
            actions=actions
        )

        # return the labels if needed
        return_labels = self.return_labels if return_labels is None else return_labels
        if return_labels:
            inputs['labels'] = inputs['input_ids'][1:].clone()

        # unsqueeze the tensors if needed
        if unsqueeze:
            # unsqueeze the tensors to make them batched
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        return inputs
    
    def process_sample(
            self,
            video_frames: torch.Tensor,
            frame_timestamps: List[float],
            text: List[str],
            text_timestamps: List[Tuple[float, float]],
            actions: List[dict] = None,
        ):
        """Preprocess an input sample.

        Turn the input text into input ids with video embedding placeholders.
        Preprocess the video frames and actions if needed.


        Args:
            video_frames (torch.Tensor): Video frames for the video clip with shape (n_frame, num_channel, height, width).
            frame_timestamps: (List[float]) List of timestamps for each video frame.
            text (List[str]): Transcripts of the video clip. List of strings, each represent a chunk of trunscript.
            text_timestamps (List[Tuple[float, float]]): Timestamps for each text chunk. List of tuples of start and end timestamps.
            actions (List[dict]): (optional) Training targets. List of actions that corresponds to each video frame.

        Returns:
            dict: Dictionary that contains the following keys:
                input_ids (torch.Tensor): Token ids that contians the video frame placeholders. (n_tokens,)
                timestamps (torch.Tensor): Timestamp for each tokens. Shape (n_tokens,).kkkkk
                video_frames (torch.Tensor): Preprocessed video frames with shape (n_frames, num_channel, height, width).
                actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to
                    each video frames. Shape (n_frames, action_dim).
        """
        # preprocess the video frames
        assert video_frames.dim() == 4, f"video_frames must have 4 dimensions, but got {video_frames.dim()}."
        video_frames = self.image_processor(video_frames)

        # Tokenize the input text and combine them with the video frame placeholders.
        if len(text) > 0:
            text_ids = self.tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']  
            # 2D list (n_chunk, n_tokens_per_chunk)
            
            # Add timestamps for all text tokens
            text_timestamps = expand_text_timestamps(text_timestamps, text_ids)
            text_ids = sum(text_ids, [])  # List[int]
        else:
            # avoid calling tokenizer with empty list, which will raise an IndexError
            text_ids = []
            text_timestamps = []

        # Make a sequence of video frame placeholders and expand frame_timestamps accordingly.
        n_frames = len(video_frames)
        frame_ids = self.frame_token_ids * n_frames  # List[int], shape (n_tokens_per_frame * n_frames,)
        frame_timestamps = np.repeat(frame_timestamps, self.frame_token_len).tolist()  # List[float], shape (n_tokens_per_frame * n_frames,)

        # Combine and sort them according to their timestamps.
        frame_id_timestamps = list(zip(frame_ids, frame_timestamps))
        text_id_timestamps = list(zip(text_ids, text_timestamps))
        all_id_timestamps = frame_id_timestamps + text_id_timestamps
        all_id_timestamps = sorted(all_id_timestamps, key=lambda x: x[1])

        input_ids, timestamps = zip(*all_id_timestamps)
        input_ids = list(input_ids)
        timestamps = list(timestamps)

        data = {
            'input_ids': torch.tensor(input_ids),
            'timestamps': torch.tensor(timestamps),
            'video_frames': video_frames,
        }
        if actions is not None:
            data['actions'] = action_dict_to_tensor(actions)
        return data

    # From transformer.models.kosmos2.processing_kosmos2.py
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # From transformer.models.kosmos2.processing_kosmos2.py
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


class PaddingCollator:
    """
    Pad the preprocessed input sequences into the same length.
    Also creates the labels for the causal language model.
    Used as a collate_fn for HuggingFace Trainer.

    Args:
        padding (Union[bool, str]): Padding strategy for input_ids. Can be one of the following:
            - True: Pad to the maximum length in the batch.
            - "longest": Pad to the longest sequence in the batch.
            - "max_length": Pad to the maximum length in the batch.
            - False: No padding.
        padding_side (str): The side on which to pad the sequence. Can be selected between ['right', 'left'].
        max_length (Optional[int]): Maximum length of the returned list and optionally padding
        pad_token_id (int): Token id for padding input_ids.
        pad_value (float): Value to pad the sequences other than input_ids.
        return_labels (bool): Whether to return the labels for input_ids.
    """
    def __init__(
            self,
            padding: Union[bool, str] = True,
            padding_side: str = "right",
            max_length: Optional[int] = None,
            pad_token_id: int = 0,
            pad_value: float = 0,
            return_labels: bool = True
    ):
        self.padding = padding
        self.padding_side = padding_side
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.return_labels = return_labels

    def __call__(self, batch):
        """
        Pad sequences into a batch.
        Pad input_ids and timestamps into a same length.
        Pad video_frames and actions into the same length.

        Args:
            batch (List[Dict]): List of preprocessed input samples. Each sample contains the following:
                input_ids (torch.Tensor): Token ids that contians the video frame placeholders. (n_tokens)
                timestamps (torch.Tensor): Timestamp for each tokens. Shape (n_tokens)
                video_frames (torch.Tensor): Preprocessed video frames with shape (n_frames, num_channel, height, width).
                actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to each video frames. Shape (n_frames, action_dim).
        """
        # get items from the batch
        input_ids = [sample['input_ids'] for sample in batch]     # list[torch.Tensor]
        timestamps = [sample['timestamps'] for sample in batch]   
        video_frames = [sample['video_frames'] for sample in batch]
        if 'actions' in batch[0]:
            actions = [sample['actions'] for sample in batch]  
        else:
            actions = None

        # check shape
        assert all(ids.dim() == 1 for ids in input_ids), "input_ids must have 1 dimension."

        # pad input_ids and timestamps: note input_ids and timestamps from a same sample have same length
        max_input_len = max(len(ids) for ids in input_ids)
        pad_length = self.max_length if self.max_length is not None else max_input_len
        input_ids, attention_mask = pad(input_ids, pad_length, padding_side=self.padding_side, pad_value=self.pad_token_id, return_attention_mask=True)
        timestamps = pad(timestamps, pad_length, pad_value=self.pad_value, padding_side=self.padding_side)

        # pad video_frames and actions
        # note that video frames are always smaller than input_ids
        max_frame_len = max(vf.size(0) for vf in video_frames)
        pad_length = self.max_length if self.max_length is not None else max_frame_len
        video_frames = pad(video_frames, pad_length, pad_value=self.pad_value, padding_side=self.padding_side)
        if actions is not None:
            actions = pad(actions, pad_length, pad_value=self.pad_value, padding_side=self.padding_side)

        # stack all items into tensors
        batch = {
            'video_frames': torch.stack(video_frames),
            'timestamps': torch.stack(timestamps),
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.tensor(attention_mask)  # list[list[int]] -> torch.Tensor
        }
        if actions is not None:
            batch['actions'] = torch.stack(actions)
        if self.return_labels:
            batch['labels'] = batch['input_ids'][:, 1:].clone()
        return batch


def pad(sequences: Union[List[List[Any]], List[torch.Tensor]],
        pad_length: int,
        pad_value: Union[int, float, torch.Tensor],
        padding_side: str = "right",
        return_attention_mask: bool = False
):
    """Pad sequences into the same length.

    Args:
        sequences: List of batch sequences to pad. If the sequences are torch.Tensor, pad first dimension.
        pad_length: Length to pad the sequences.
        pad_value: Value to pad the sequences.
        padding_side: Side to pad the sequences. Can be 'right' or 'left'.
        return_attention_mask: Whether to create attention mask.

    Returns:
        Union[List[List[Any]], List[torch.Tensor]]: Padded sequences.
        List[List[int]]: Attention mask if return_attention_mask is True.
    """
    assert isinstance(sequences, list), "sequences must be a list of sequences."

    # create attention mask first
    if return_attention_mask:
        if padding_side == "right":
            attention_mask = [
                [1] * min(len(seq), pad_length) + [0] * max(pad_length - len(seq), 0) for seq in sequences
            ]
        elif padding_side == "left":
            attention_mask = [
                [0] * max(pad_length - len(seq), 0) + [1] * min(len(seq), pad_length) for seq in sequences
            ]
        else:
            raise ValueError(f"padding_side must be one of ['right', 'left'], but got {padding_side}.")

    if isinstance(sequences[0], list):
        if padding_side == "right":
            sequences = [seq + [pad_value] * (pad_length - len(seq)) for seq in sequences]
        elif padding_side == "left":
            sequences = [[pad_value] * (pad_length - len(seq)) + seq for seq in sequences]
        else:
            raise ValueError(f"padding_side must be one of ['right', 'left'], but got {padding_side}.")
        
    elif isinstance(sequences[0], torch.Tensor):
        dim = sequences[0].dim()
        if padding_side == "right":
            # pad first dimension on right
            # pad (0, 0, ...., 0, pad_length - len(seq))
            sequences = [
                F.pad(seq, tuple([0] * (2*dim - 1) + [pad_length - len(seq)]), value=pad_value) for seq in sequences
            ]
        elif padding_side == "left":
            # pad first dimension on left
            # pad (0, 0, ...., pad_length - len(seq), 0)
            sequences = [
                F.pad(seq, tuple([0] * (2*dim - 2) + [pad_length - len(seq)] + [0]), value=pad_value) for seq in sequences
            ]
        else:
            raise ValueError(f"padding_side must be one of ['right', 'left'], but got {padding_side}.")
    else:
        raise ValueError(f"sequences must be a list of list or list of torch.Tensor. Got {type(sequences[0])}.")
    
    if return_attention_mask:
        return sequences, attention_mask
    else:
        return sequences


def expand_text_timestamps(
        timestamps: Union[torch.Tensor, List[Tuple[float, float]]],
        text_ids: List[List[int]]) -> List[float]:
    """
    Expand the original text_timesmtamps, which only shows the start and end time of each text chunk,
    to the pseudo timestamps for each token.

    Args:
        timestamps: List of tuples of (start_time, end_time) for each text chunk.
            Or a tensor with shape (n_chunks, 2).
        text_ids: List of text token ids for each text chunk.

    Returns:
        list: The expanded timestamps that corresponds to each tokens.
    """
    assert len(timestamps) == len(text_ids), "timestamps and text_ids must have the same length."
    expanded_timestamps = []

    # TODO: Faster implementation
    for i, (t_s, t_e) in enumerate(timestamps):
        n_tokens = len(text_ids[i])
        expanded_timestamps += np.linspace(t_s, t_e, n_tokens).tolist()
    
    return expanded_timestamps


ACTION_KEYS = [
    'attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 
    'drop', 'inventory', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 
    'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 'camera'
]
ACTION_KEYS_TENSOR = [
    'attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 
    'drop', 'inventory', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 
    'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 'camera_0', 'camera_1'
]


def action_dict_to_tensor(action: Dict[str, Union[List[int], List[List[float]]]]):
    """
    Turn an action dictionary that contains actions of a clip
    into a tensor.

    Args:
        action: Dictionary that maps action names to their values.
            The values contains the action key press or camera movements
            for each video frames.
    Returns:
        torch.Tensor: Shape (n_frames, action_dim)
    """
    action_gathered = []
    for key in ACTION_KEYS:
        if key == 'camera':
            camera_0 = [v[0] for v in action[key]]
            camera_1 = [v[1] for v in action[key]]
            action_gathered.append(camera_0)
            action_gathered.append(camera_1)
        else:
            action_gathered.append(action[key])
    action_tensor = torch.tensor(action_gathered)  # (action_dim, n_frames)
    action_tensor = action_tensor.transpose(0, 1)  # (n_frames, action_dim)
    return action_tensor


def action_tensor_to_dict(action: torch.Tensor):
    """
    Turn an action tensor into a dictionary.
    This is the reverse of `action_dict_to_tensor`.

    Args:
        action: Tensor with shape (n_frames, action_dim).
    Returns:
        dict: Dictionary that maps action names to their values
        at each video frames.
    """
    action_dict = {}
    for idx, key in enumerate(ACTION_KEYS_TENSOR):
        action_dict[key] = action[:, idx].tolist()
    # gather camera movements into a single list
    camera_0 = action_dict.pop('camera_0')
    camera_1 = action_dict.pop('camera_1')
    action_dict['camera'] = [[c0, c1] for c0, c1 in zip(camera_0, camera_1)]

    return action_dict


