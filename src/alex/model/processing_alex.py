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
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 image_processor: Optional[Callable] = None,
                 frame_start_token: str = "<frame/>",
                 frame_emb_token: str = "<frame>", 
                 frame_end_token: str = "</frame>",
                 frame_emb_len: int = 1,
                 default_fps: int = 30
                 ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.frame_start_token = frame_start_token
        self.frame_token = frame_emb_token
        self.frame_end_token = frame_end_token
        self.frame_emb_len = frame_emb_len
        self.default_fps = default_fps

        # id: int, token: str
        self.pad_token_id = tokenizer.pad_token_id

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
        

    def collate_fn(
            self,
            batch
    ):
        """
        Collate function for transformers Trainer class.
        Apply preprocessing and padding to the input samples.

        Args:
            batch: List of samples to collate.
        """
        return self.__call__(
            video_frames=[sample['video_frames'] for sample in batch],
            frame_timestamps=[sample['frame_timestamps'] for sample in batch],
            text=[sample['text'] for sample in batch],
            text_timestamps=[sample['text_timestamps'] for sample in batch],
            actions=[sample['actions'] for sample in batch],
        )
    
    def __call__(
            self,
            video_frames: Union[List[torch.Tensor], torch.Tensor],
            frame_timestamps: List[List[float]],
            text: List[List[str]],
            text_timestamps: List[List[Tuple[float, float]]],
            actions: List[List[dict]] = None,
            padding: Union[bool, str] = True,
            padding_side: str = "right",
            max_length: Optional[int] = None
        ):
        """
        Process input samples and make a batch. 

        TODO: this method is not compatible with a single input sample.

        Args:
            video_frames (List[torch.Tensor]): List of video frames for each video clip.
                Each element is video frames in a video clip with shape (n_frames, num_channel, height, width).
            frame_timestamps (List[List[float]]): List of timestamps for each video frame.
                Each element is a list of timestamps for each video clip.
            text (List[List[str]]): List of transcripts of the video clips.
                Each element is a list of strings, each represent a chunk of trunscript.
            text_timestamps (List[List[Tuple[float, float]]]): List of timestamps for each text chunk.
                Each element is a list of tuples of start and end timestamps.
            actions (List[List[dict]]): (optional) Training targets. List of actions that corresponds to each video frame.
            padding (Union[bool, str]): Padding strategy for input_ids. Can be one of the following:
                - True: Pad to the maximum length in the batch.
                - "longest": Pad to the longest sequence in the batch.
                - "max_length": Pad to the maximum length in the batch.
                - False: No padding.
            padding_side (str): The side on which to pad the sequence. Can be selected between ['right', 'left'].
            max_length (Optional[int]): Maximum length of the returned list and optionally padding length.

        Returns:

        """
        # check if inputs are batched or a single sample
        if isinstance(video_frames, torch.Tensor) and video_frames.dim() == 4:
            # a single sample
            processed = self.process_sample(
                video_frames=video_frames,
                frame_timestamps=frame_timestamps,
                text=text,
                text_timestamps=text_timestamps,
                actions=actions
            )
            processed = [processed]

        else:
            # Process each sample
            processed = [
                self.process_sample(
                    video_frames=video_frames[i],
                    frame_timestamps=frame_timestamps[i],
                    text=text[i],
                    text_timestamps=text_timestamps[i],
                    actions=actions[i] if actions is not None else None
                )
                for i in range(len(video_frames))
            ]

        # stack all items into a list
        input_ids = [data['input_ids'] for data in processed]    # List[List[int]]
        timestamps = [data['timestamps'] for data in processed]  # List[List[float]]
        video_frames = [data['video_frames'] for data in processed] # list of tensors with shape (n_frames, num_channel, height, width)
        if actions is not None:
            actions = [data['actions'] for data in processed]
        
        if padding:
            # pad input_ids and timestamps: note input_ids and timestamps from a same sample have same length
            max_input_len = max(len(ids) for ids in input_ids)
            pad_length = max_length if max_length is not None else max_input_len
            input_ids, attention_mask = self.pad(input_ids, pad_length, padding_side=padding_side, pad_value=self.pad_token_id, return_attention_mask=True)
            timestamps = self.pad(timestamps, pad_length, pad_value=0, padding_side=padding_side)

            # pad video_frames and actions
            max_frame_len = max(vf.size(0) for vf in video_frames)
            video_frames = self.pad(video_frames, max_frame_len, pad_value=0, padding_side=padding_side)
            if actions is not None:
                actions = self.pad(actions, max_frame_len, pad_value=0, padding_side=padding_side)

        # stack all items into tensors
        batch = {
            'video_frames': torch.stack(video_frames),
            'timestamps': torch.tensor(timestamps),
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask) if padding else None,
        }
        if actions is not None:
            batch['actions'] = torch.stack(actions)
        return batch
    
    def pad(
            self,
            sequences: Union[List[List[Any]], List[torch.Tensor]],
            pad_length: int,
            pad_value: Union[int, float, torch.Tensor],
            padding_side: str = "right",
            return_attention_mask: bool = False
    ):
        """Pad sequences into the same length.

        Args:
            sequences: List of sequences to pad. If the sequences are torch.Tensor, pad first dimension.
            pad_length: Length to pad the sequences.
            pad_value: Value to pad the sequences.
            padding_side: Side to pad the sequences. Can be 'right' or 'left'.
            return_attention_mask: Whether to create attention mask.

        Returns:
            Union[List[List[Any]], List[torch.Tensor]]: Padded sequences.
            List[List[int]]: Attention mask if return_attention_mask is True.
        """
        # create attention mask first
        if return_attention_mask:
            if padding_side == "right":
                attention_mask = [[1] * len(seq) + [0] * (pad_length - len(seq)) for seq in sequences]
            elif padding_side == "left":
                attention_mask = [[0] * (pad_length - len(seq)) + [1] * len(seq) for seq in sequences]
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
                sequences = [
                    F.pad(seq, tuple([0] * (dim - 1) + [pad_length - len(seq)]), value=pad_value) for seq in sequences
                ]
            elif padding_side == "left":
                sequences = [
                    F.pad(seq, tuple([0] * (dim - 2) + [pad_length - len(seq)] + [0]), value=pad_value) for seq in sequences
                ]
            else:
                raise ValueError(f"padding_side must be one of ['right', 'left'], but got {padding_side}.")
        else:
            raise ValueError(f"sequences must be a list of list or list of torch.Tensor. Got {type(sequences[0])}.")
        
        if return_attention_mask:
            return sequences, attention_mask
        else:
            return sequences

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
                input_ids (List[int]): Token ids that contians the video frame placeholders. 
                timestamps (List[int]): Timestamp for each tokens. Shape (n_tokens,).
                video_frames (torch.Tensor): Preprocessed video frames with shape (n_frames, num_channel, height, width).
                actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to
                    each video frames. Shape (n_frames, action_dim).
        """
        # preprocess the video frames
        assert video_frames.dim() == 4, f"video_frames must have 4 dimensions, but got {video_frames.dim()}."
        video_frames = self.image_processor(video_frames)

        # Tokenize the input text and combine them with the video frame placeholders.
        text_ids = self.tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']  
        # 2D list (n_chunk, n_tokens_per_chunk)
        
        # Add timestamps for all text tokens
        text_timestamps = expand_text_timestamps(text_timestamps, text_ids)
        text_ids = sum(text_ids, [])  # List[int]

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

        # Preprocess the actions.
        if actions is not None:
            actions = action_dict_to_tensor(actions)

        # 4. Return the results
        data = {
            'input_ids': input_ids,
            'timestamps': timestamps,
            'video_frames': video_frames,
            'actions': actions,
        }
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


