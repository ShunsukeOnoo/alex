from typing import List, Optional, Tuple, Union
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
    """
    def __init__(self, tokenizer, image_processor = None, frame_token: str = "<frame>"):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.frame_token = frame_token

        # TODO: Add frame token to the tokenizer
        # self.tokenizer.add_special_tokens({
        #     "additional_special_tokens": [self.frame_token]}
        #     )

    def check_input(self, video_frames, frame_timestamps, actions, text, text_timestamps):
        """
        Check if the input is valid.
        """
        # TODO: Determine input combinations: see the docstring of __call__
        if text is None and video_frames is None:
            raise ValueError("Either text or video_frames must be provided.")
        
        # TODO: If timestamps are provided, check the length.
        # TODO: If not, set the pseudo timestamps.
        if video_frames is not None and frame_timestamps is not None:
            assert len(video_frames) == len(frame_timestamps),\
                "video_frames and frame_timestamps must have the same length."
        if text is not None and text_timestamps is not None:
            assert len(text) == len(text_timestamps),\
                "text and text_timestamps must have the same length."
            
        # TODO: Check the format of actions
        if actions is not None:
            assert isinstance(actions, list), "actions must be a list."
            assert len(actions) == len(video_frames), \
                f"actions must have the same length as video_frames but got {len(actions)} and {len(video_frames)}."
    
    def __call__(
            self,
            video_frames: List[Image.Image] = None,
            frame_timestamps: List[float] = None,
            actions: List[dict] = None,
            text: List[str] = None,
            text_timestamps: List[str] = None,
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
            text_timestamps: (optional) List of timestamps for each chunk.
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
                to a video frame. Shape (n_tokens,). -1 for text tokens. Integer larger or 
                equal to 0 for video frames. The number corresponds to the index of the video frame.
            actions (torch.Tensor): (Optional) Preprocessed actions that corresponds to
                each video frames. Shape (n_frames, action_dim).
        """
        # 0. Check if the input is valid
        # Check the missing inputs
        self.check_input(video_frames, frame_timestamps, actions, text, text_timestamps)

        # 1. Tokenize the input text and combine them with the video frame placeholders.
        # TODO: Is this padding stragety common across models?
        text_ids = self.tokenizer(text, padding='do_not_pad', return_tensors=None)['input_ids']
        # 2d list (n_chunk, n_tokens_per_chunk)
        text_ids = sum(text_ids, [])  # List[int]
        # Add timestamps for all text tokens
        text_timestamps = expand_text_timestamps(text_timestamps, text_ids)

        # Make a sequence of video frame placeholders.
        # TODO: We don't need to call the tokenizer for this.
        # TODO: Single frame might correspond to multiple tokens.
        frame_tokens = self.frame_token * video_frames.shape[0]
        frame_ids = self.tokenizer(frame_tokens, padding='do_not_pad', return_tensors=None)['input_ids']
        # frame_ids: lst[int]
        
        # Combine and sort them according to their timestamps.
        input_ids, timestamps, video_mask = combine_and_sort_ids(text_ids, text_timestamps, frame_ids, frame_timestamps)
        # input_ids and timestamps, shape (n_tokens,)

        # 2. Create masks.
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
            'attention_mask': attention_mask,
            'video_frame_mask': video_mask
        }
        return data
    

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


def expand_text_timestamps(text_timestamps: List[float], text_ids: torch.Tensor) -> List[float]:
    """
    Expand the original text_timesmtamps, which only shows the start and end time of each text chunk,
    to the pseudo timestamps for each token.

    TODO: Implement this function.
    TODO: Use sophisticated method.
    """
    pass


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
        Tuple[torch.Tensor, torch.Tensor]: Combined and sorted token ids and timestamps.
    """
    text_ids = [(text_ids[i], text_timestamps[i]) for i in range(len(text_ids))]
    frame_ids = [(frame_ids[i], frame_timestamps[i]) for i in range(len(frame_ids))]
    text_ids = text_ids + frame_ids
    text_ids.sort(key=lambda x: x[1])
    input_ids, timestamps = zip(*text_ids)
    input_ids = torch.tensor(text_ids)
    timestamps = torch.tensor(timestamps)
    
    # TODO: Create video mask
    video_mask = None
    return input_ids, timestamps, video_mask





# TODO: turn the following two functions into a single function
def pad_input_ids(input_ids: List[torch.Tensor], pad_token: int, max_len: int = None) -> torch.Tensor:
    """
    Pad sequences of input ids to specified length and turn them into a single tensosrs. 
    Sequences longer than `max_len` will be cut.

    Args:
        input_ids (List[torch.Tensor]): List of input ids with shape (n_tokens) each.
        max_len (int): Length of the resulting tensor.

    Returns:
        torch.Tensor: Batch of input ids. Shape (batch_size, max_len)
    """
    batch = nn.utils.rnn.pad_sequences(input_ids, batch_first=True, padding_value=pad_token)
    # batch: torch.Tensor with shape (batch_size, l), 
    # where l is the maximum length of the input_ids
    if max_len is not None:
        batch = batch[:, :max_len]
    return batch


def pad_videos(video_frames: List[torch.Tensor], pad_value: float = 0.0):
    """
    Pad sequences of video frames to the same length and turn them into a batch.

    Args:
        video_frames: List of video frames with shape (n_frames, n_channels, height, width)
        pad_value: Value to pad the video frames with.
    Returns:
        torch.Tensor: Shape (batch_size, n_frames, n_channels, height, width)
    """
    batch = nn.utils.rnn.pad_sequence(
        video_frames, batch_first=True, padding_value=pad_value)
    return batch


def action_dict_to_list(action: dict) -> List[float]:
    """
    Flatten an action dictionary into a 1d list.
    """
    action_list = [
        value for value in action.values()  # List[List[float]]
    ]
    action_list = sum(action_list, [])  # List[float]
    return action_list


def action_dicts_to_tensor(actions: List[dict]) -> torch.Tensor:
    """
    Turn a list of action dictionaries into a tensor.

    Args:
        actions: List of action dictionaries. Each dictionary maps
            action names to their corresponding values (list).
    Returns:
        torch.Tensor: Shape (batch_size, action_dim)
    """
    actions = [action_dict_to_list(action) for action in actions]
    # List[List[float]]
    actions = torch.tensor(actions)
    return actions


# TODO: replace with the real action format
ACTION_FORMAT = {
    'jump': [0],
    'left': [0],
    'right': [0],
}


def action_list_to_dict(action_list: List[float], action_format: dict = ACTION_FORMAT) -> dict:
    """
    Turn a 1d list of actions (one sample) into a dictionary.
    """
    # check the format
    action_dim = sum(len(lst) for lst in action_format.values())
    assert len(action_list) == action_dim, 'The action list does not match the action format.'

    # TODO: Implement more efficient way
    # convert the list to a dictionary
    idx = 0
    action_dict = copy.deepcopy(action_format)
    for key, value in action_format.items():
        n_dim = len(value)
        action_dict[key] = action_list[idx: idx + n_dim]
        idx += n_dim
    return action_dict


def action_tensor_to_dicts(actions: torch.Tensor, action_format: dict = ACTION_FORMAT) -> List[dict]:
    """
    Turn a tensor of actions into a list of action dictionaries.
    """
    actions = actions.tolist()
    action_dicts = [action_list_to_dict(action, action_format) for action in actions]
    return action_dicts