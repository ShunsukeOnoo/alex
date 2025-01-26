from typing import List
import copy
import numpy as np
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


def insert_video_embeddings(
        text_embeddings: torch.Tensor,
        video_embeddings: torch.Tensor,
        video_frame_mask: torch.Tensor,
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
    # TODO: Improve the efficiency
    # Iterate over the batch
    # Simple but not efficient
    embeddings = torch.clone(text_embeddings)
    for i in range(embeddings.size(0)):
        n_frames = video_frame_mask[i].sum().item()
        embeddings[i, video_frame_mask[i]] = video_embeddings[i, :n_frames]
    
    return embeddings


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