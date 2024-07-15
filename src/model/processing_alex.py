from typing import List, Optional, Tuple, Union, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
# TODO: Use the ProcessorMixin?
# from transformers.processing_utils import ProcessorMixin


class AlexProcessor:
    """
    Processor for the Alex model. It handles
    1. tokenization of the input text,
    2. preprocessing of the video frames.
    3. aligning the video frames with the text tokens.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for the text input.
        image_processor (callable, optional): Image processor for the video frames.
            If None, the video frames are not processed.
        frame_token (str): Placeholder for the video frames.
        frame_end_token (str): End token for the video frames.
        frame_emb_len (int): Length of the frame embeddings.
            The overall placeholder for a single frame will be frame_token * (frame_emb_len - 1) + frame_end_token.
    """
    def __init__(self, 
                 tokenizer, 
                 image_processor = None, 
                 frame_token: str = "<frame>", 
                 frame_end_token: str = "</frame>",
                 frame_emb_len: int = 1,
                 default_fps: int = 30
                 ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.frame_token = frame_token
        self.frame_end_token = frame_end_token
        self.default_fps = default_fps

        # Add special tokens
        self.num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': [frame_token, frame_end_token]})
        self.frame_emb_len = frame_emb_len
        self.frame_token_str = frame_token * (frame_emb_len - 1) + frame_end_token
        self.frame_tokens = tokenizer.encode(self.frame_token_str)
        self.frame_last_emb_token = tokenizer.convert_tokens_to_ids(frame_end_token)


    def check_input(
            self, 
            video_frames: List[Image.Image], 
            frame_timestamps: List[float], 
            actions: List[dict], 
            text: List[str], 
            text_timestamps: List[Tuple[float, float]]
            ):
        """
        Check if the input is valid, and create psuedo timestamps if needed.
        """
        # TODO: Determine input combinations: see the docstring of __call__
        input_types = []
        if text is not None:
            input_types.append('text')
        if video_frames is not None:
            input_types.append('video')
        if len(input_types) == 0:
            raise ValueError("Either text or video_frames must be provided.")
        if actions is not None:
            input_types.append('actions')
        
        if 'text' in input_types:
            if text_timestamps is not None:
                assert len(text) == len(text_timestamps),\
                    "text and text_timestamps must have the same length."
            else:
                text_timestamps = [(0, 0) for _ in text]
        if 'video' in input_types:
            if frame_timestamps is not None:
                assert len(video_frames) == len(frame_timestamps),\
                    "video_frames and frame_timestamps must have the same length."
            else:
                frame_timestamps = np.arange(0, len(video_frames), 1/self.default_fps).tolist()
            
        # TODO: Check the format of actions
        if actions is not None:
            assert isinstance(actions, list), "actions must be a list."
            assert len(actions) == len(video_frames), \
                f"actions must have the same length as video_frames but got {len(actions)} and {len(video_frames)}."
            
        return text_timestamps, frame_timestamps
    
    def __call__(
            self,
            video_frames: List[Image.Image] = None,
            frame_timestamps: List[float] = None,
            actions: List[dict] = None,
            text: List[str] = None,
            text_timestamps: List[Tuple[float, float]] = None,
        ):
        """
        Preprocess an input sample.

        TODO: Come up with more efficient way.
        TODO: Accept batch inputs.

        I consider multiple input combinations:
        - text only: For text generation task.
        - video_frames only: For action generation without text conditioning.
        - text and video_frames : For action generation with text conditioning.
        
        Timestamps for each inputs are recommended but not required. If not provided,
        the timestamps will be set as follows:
        TODO: consider default timestamps.
        - If there is text, the timestamps will be set as the start time of the sample.
        - If there are video_frames, the timestamps will be set using default fps.

        Process:
        1. Tokenize the input text and combine them with the video frame placeholders.
        2. Create masks.
        3. Preprocess the video frames.
        4. Preprocess the actions.

        Args:
            text: (optional) Transcripts of the video clip. List of strings, each represent a chunk of trunscript.
            text_timestamps: (optional) Tuple of start and end timestamps for each text chunk.
                Must be the same length as the text.
            video_frames: (optional) List of video frames in the video clip.
            frame_timestamps: (optional) List of timestamps for the video frames.
                Must be the same length as the video_frames.
            actions: (optional) Training targets. List of actions to be taken. Each action is a dictionary.

        Returns a dict with following keys and values:
            input_ids (torch.Tensor): Token ids that contians the video frame placeholders.
                Shape (n_tokens,). 
                Placeholder ids are the same for all the video frames. Each video frame
                may corresponds to multiple placeholders when frame embeddings are more than
                one token.
            video_frames (torch.Tensor): Preprocessed video frames. 
                Shape (n_frames, num_channel, height, width).
            timestamp (torch.Tensor): Timestamp for each tokens. Shape (n_tokens,).
            video_mask (torch.Tensor): Indicates a position in input_ids that corresponds
                to a video frame. Shape (n_tokens,). 0 for text tokens and 1 for video frames.
            actin_target_mask (torch.Tensor): Indicates the position that predicts the action.
                Shape (n_tokens,). 1 for the position that predicts the action. 0 for the others.
                For each video frame, the lsat embedding position will be 1.
            text_target_mask (torch.Tensor): Indicates the position that predicts the text.
                Shape (n_tokens,). 1 for the position that predicts the text. 0 for the others.
                Positions whose next token is a text token will be 1.
            actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to
                each video frames. Shape (n_frames, action_dim).
        """
        # 0. Check if the input is valid
        # Check the missing inputs
        text_timestamps, frame_timestamps = self.check_input(video_frames, frame_timestamps, actions, text, text_timestamps)

        # 1. Tokenize the input text and combine them with the video frame placeholders.
        text_ids = self.tokenizer.encode(text)  # List[List[int]], 2d list (n_chunk, n_tokens_per_chunk)
        # Add timestamps for all text tokens
        text_timestamps = expand_text_timestamps(text_timestamps, text_ids)
        text_ids = sum(text_ids, [])  # List[int]

        # Make a sequence of video frame placeholders.
        n_frames = video_frames.shape[0]
        frame_ids = self.frame_tokens * n_frames  # List[int]
        # frame_ids: lst[int]
        
        # Combine and sort them according to their timestamps.
        input_ids, timestamps, is_video = combine_and_sort_ids(text_ids, text_timestamps, frame_ids, frame_timestamps)
        # input_ids and timestamps, shape (n_tokens,)

        # 2. Create masks.
        video_mask, text_target_mask, action_target_mask = self.create_masks(input_ids, is_video)
        attention_mask = torch.tensor([1] * len(input_ids))

        # 3. Preprocess the video frames if needed.
        if self.image_processor is not None:
            video_frames = self.image_processor(video_frames)

        # 4. Preprocess the actions.
        if actions is not None:
            actions = action_dicts_to_tensor(actions)

        # 4. Return the results
        data = {
            'input_ids': input_ids,
            'video_frames': video_frames,
            'timestamps': timestamps,
            'actions': actions,
            'action_target_mask': action_target_mask,
            'text_target_mask': text_target_mask,
            'attention_mask': attention_mask,
            'video_frame_mask': video_mask
        }
        return data
    
    def create_mask(self, input_ids: torch.Tensor, is_video: List[bool]):
        """
        Create masks for the input_ids.

        Args:
            input_ids (torch.Tensor): Token ids with shape (n_tokens,).
            is_video (List[bool]): Indicates whether the token is a video frame or not.

        Returns: Tensors with shape (n_tokens,) and dtype torch.int8.
            video_mask (torch.Tensor): Indicates the position of the video frame embeddings.
            text_target_mask (torch.Tensor): Indicates the position that predicts the text.
            action_target_mask (torch.Tensor): Indicates the position that predicts the action.
        """
        video_mask = torch.tensor(is_video, dtype=torch.int8)

        text_target_mask = 1 - torch.tensor(is_video, dtype=torch.int8)
        text_target_mask = text_target_mask[1:]  # exclude the first token
        text_target_mask = torch.cat([text_target_mask, torch.tensor([0], dtype=torch.int8)])  # add a zero to the end

        action_target_mask = input_ids == self.frame_last_emb_token
        action_target_mask = action_target_mask.to(torch.int8)

        return video_mask, text_target_mask, action_target_mask

    # Copied from transformer.models.kosmos2.processing_kosmos2.py
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformer.models.kosmos2.processing_kosmos2.py
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


def expand_text_timestamps(timestamps: List[Tuple[float, float]], text_ids: List[List[int]]) -> List[float]:
    """
    Expand the original text_timesmtamps, which only shows the start and end time of each text chunk,
    to the pseudo timestamps for each token.

    Args:
        timestamps: List of tuples of (start_time, end_time) for each text chunk.
        text_ids: List of text token ids for each text chunk.

    Returns:
        List of timestamps for each text token.
    """
    assert len(timestamps) == len(text_ids), "timestamps and text_ids must have the same length."
    expanded_timestamps = []

    for i, (start, end) in enumerate(timestamps):
        n_tokens = len(text_ids[i])
        chunk_timestamps = np.linspace(start, end, n_tokens)
        expanded_timestamps.append(list(chunk_timestamps))

    return expanded_timestamps


def combine_and_sort_ids(
        text_ids: List[int], 
        text_timestamps: List[float],
        frame_ids: List[int],
        frame_timestamps: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine and sort the text ids and frame ids according to their timestamps.
    TODO: Add shapes of the inputs and outputs.
    TODO: Create video mask at the same time.

    Args:
        text_ids: List of token ids for the text.
        text_timestamps: List of timestamps for each token.
        frame_ids: List of token ids for the video frame placeholders.
        frame_timestamps: List of timestamps for each video frame.
    Returns:
        input_ids (torch.Tensor): Combined token ids.
        timestamps (torch.Tensor): Combined timestamps.
        is_video (List[bol]): Indicates whether the token is a video frame or not.
    """
    # List of tuple (id, timestamp, is_video)
    text_ids = [(text_ids[i], text_timestamps[i], False) for i in range(len(text_ids))]
    frame_ids = [(frame_ids[i], frame_timestamps[i], True) for i in range(len(frame_ids))]

    text_ids = text_ids + frame_ids
    text_ids.sort(key=lambda x: x[1])
    
    input_ids, timestamps, is_video = zip(*text_ids)
    input_ids = torch.tensor(text_ids)
    timestamps = torch.tensor(timestamps)
    
    return input_ids, timestamps, is_video


# TODO: replace with the real action format
ACTION_FORMAT = {
    'jump': [0],
    'left': [0],
    'right': [0],
}
# TODO: Fill this according to hte IDM
ACTION_KEYS = []
ACTION_KEYS_MAP = {}  # maps action names to their idx in the tensor

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
    action_tensor = torch.tensor(action_gathered)
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
    for key, idx in ACTION_KEYS_MAP.items():
        action_dict[key] = action[idx]
    # gather camera movements into a single list
    camera_0 = action_dict.pop('camera_0')
    camera_1 = action_dict.pop('camera_1')
    action_dict['camera'] = [[c0, c1] for c0, c1 in zip(camera_0, camera_1)]

    return action_dict


