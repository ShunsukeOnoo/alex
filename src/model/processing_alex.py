from typing import List, Optional, Tuple, Union
import numpy as np
import PIL
from PIL import Image
import torch
# TODO: Use the ProcessorMixin
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
        if text is None and video_frames is None:
            raise ValueError("Either text or video_frames must be provided.")
        if video_frames is not None and frame_timestamps is not None:
            assert len(video_frames) == len(frame_timestamps),\
                "video_frames and frame_timestamps must have the same length."
        if text is not None and text_timestamps is not None:
            assert len(text) == len(text_timestamps),\
                "text and text_timestamps must have the same length."
    
    def __call__(
            self,
            video_frames: List[PIL.Image] = None,
            frame_timestamps: List[float] = None,
            actions: List[dict] = None,
            text: List[str] = None,
            text_timestamps: List[str] = None,
            padding: bool = False,
            max_length: int = None,
            return_tensors: str = 'pt'
        ):
        """
        Preprocess input

        Process:
        1. Tokenize the input text and combine them with the video frame placeholders.
        2. Preprocess the video frames.
        3. Preprocess the actions.

        Args:
            text (List[str]): List of text to be tokenized.
            text_timestamps (List[float]): List of timestamps for the text tokens. 
                Must be the same length as the text.
            video_frames(List[PIL.Image]): List of video frames to be preprocessed.
            frame_timestamps (List[float]): List of timestamps for the video frames.
                Must be the same length as the video_frames.
            actions (List[AlexActions]): List of actions to be taken.

        Returns:
            Dict[str, torch.Tensor]]: Dictionary that have follosing keys annd values:
                input_ids (torch.Tensor): Shape (batch_size, n_tokens)
                # TODO: What if each batch element has different number of frames?
                video_frames (torch.Tensor): Processed video frames. Shape (batch_size, n_frames, n_channels, height, width)
                timestamps (torch.Tensor): Shape (batch_size, n_tokens)
                video_frame_mask
                actions
        """
        # 0. Check if the input is valid
        # Check the missing inputs
        self.check_input(video_frames, frame_timestamps, actions, text, text_timestamps)
        
        # TODO: Extend this to handle batch inputs.
        # Let's assume the input is one sample for now.

        # Let's try to combine str first and then tokenize it.
        # Create one string that contains all the text and frame placeholders.
        frame_tokens = [self.frame_token] * video_frames.shape[0]
        frame_tokens = [
            (frame_tokens[i], frame_timestamps[i]) for i in range(len(frame_tokens))
        ]
        # For now, only use the start time of the text chunk.
        text = [
            (text[i], text_timestamps[i][0]) for i in range(len(text))
        ]
        text = text + frame_tokens
        text.sort(key=lambda x: x[1])
        text, timestamps = [t[0] for t in text], [t[1] for t in text]
        text = " ".join(text)

        # tokenize the text
        input_ids = self.tokenizer(text, padding=padding, max_length=max_length, return_tensors=return_tensors)

        # Since there will be more tokens than len(timestamps), 
        # We have to expand timestamps
        timestamps = torch.zeros_like(input_ids)


        # 1. Tokenize the input text and combine them with the video frame placeholders.
        # Another method: tokenize each text chunk and then concatenate them.
        text_ids = self.tokenizer(text, padding=None, return_tensors=None)['input_ids']
        # 2d list (n_chunk, n_tokens_per_chunk)
        # Add timestamps for all the tokens
        timestamps = []
        for i, chunk in enumerate(text_ids):
            s, e = text_timestamps[i]
            t = np.linspace(s, e, len(chunk)).tolist()
            timestamps.extend(t)
        # combine the text_ids
        text_ids = sum(text_ids, [])
        text_ids = [(text_ids[i], timestamps[i]) for i in range(len(text_ids))]

        # make the video frame placeholders
        frame_tokens = self.frame_token * video_frames.shape[0]
        frame_ids = self.tokenizer(frame_tokens, padding=None, return_tensors=None)['input_ids']
        frame_ids = [(frame_ids[i], frame_timestamps[i]) for i in range(len(frame_ids))]

        # combine and sort them
        text_ids = text_ids + frame_ids
        text_ids.sort(key=lambda x: x[1])
        text_ids, timestamps = zip(*text_ids)
        text_ids = torch.tensor(text_ids)
        timestamps = torch.tensor(timestamps)
        # Now we have the input_ids and timestamps


        # 2. Preprocess the video frames.
        if self.image_processor is not None:
            video_frames = self.image_processor(video_frames)

        # 3. Preprocess the actions.
        action_tensor = action_to_tensor(actions)

        # 4. Return the results
        data = {
            'input_ids': input_ids,
            'video_frames': video_frames,
            'timestamps': timestamps,
            'actions': action_tensor,
            # 'attention_mask': attention_mask
            # 'video_frame_mask': video_frame_mask
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