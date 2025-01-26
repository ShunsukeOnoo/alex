"""
Util for model and preprocessing.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

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