"""
Simple test for processing functions.
"""
import os
import sys
print(os.getcwd())
print(sys.path)

import numpy as np 
import torch
from transformers import AutoTokenizer
from processing_alex import (
    expand_text_timestamps, combine_and_sort_ids, 
    action_dict_to_tensor, action_tensor_to_dict,
    AlexProcessor
)


timestamps = expand_text_timestamps(
    [[0, 1], [3, 5]],
    [[1, 2], [3, 4, 5]]
)
assert timestamps == [0.0, 1.0, 3.0, 4.0, 5.0]


input_ids, timestamps, is_video = combine_and_sort_ids(
    text_ids=[0, 1, 2, 3, 4, 5],
    text_timestamps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    frame_ids=[77, 78, 77, 78],
    frame_timestamps=[0.15, 0.15, 0.35, 0.35]
)
assert torch.equal(input_ids, torch.tensor([0, 1, 77, 78, 2, 3, 77, 78, 4, 5]))
assert torch.allclose(timestamps, torch.tensor([0.0, 0.1, 0.15, 0.15, 0.2, 0.3, 0.35, 0.35, 0.4, 0.5]), rtol=1e-5)
assert is_video == [False, False, True, True, False, False, True, True, False, False]


# Sample input dictionary
action_dict = {
    'attack': [1, 0, 1],
    'back': [0, 0, 0],
    'forward': [0, 1, 1],
    'jump': [0, 1, 0],
    'left': [0, 0, 1],
    'right': [1, 1, 0],
    'sneak': [0, 0, 0],
    'sprint': [1, 0, 1],
    'use': [0, 1, 1],
    'drop': [0, 0, 0],
    'inventory': [1, 1, 1],
    'hotbar.1': [0, 0, 0],
    'hotbar.2': [0, 0, 0],
    'hotbar.3': [0, 0, 0],
    'hotbar.4': [0, 0, 0],
    'hotbar.5': [0, 0, 0],
    'hotbar.6': [0, 0, 0],
    'hotbar.7': [0, 0, 0],
    'hotbar.8': [0, 0, 0],
    'hotbar.9': [0, 0, 0],
    'camera': [[0.5, 0.2], [0.1, -0.3], [0.0, 0.0]]
}

# Expected output tensor
action_tensor = torch.tensor([
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.2],
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, -0.3],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0]
])

# test action conversion functions
output_tensor = action_dict_to_tensor(action_dict)
assert torch.equal(output_tensor, action_tensor), "Test case failed!"

output_dict = action_tensor_to_dict(action_tensor)
for key in output_dict:
    assert np.allclose(output_dict[key], action_dict[key]), "Test case failed!"


# Test processor class
tokenizer = AutoTokenizer.from_pretrained('gpt2')
processor = AlexProcessor(tokenizer)

video_frames = torch.randn(3, 3, 224, 224)
frame_timestamps = np.array([1, 2, 3])
text = ["Let's get some wood.", "Then we need an axe."]
text_timestamps = torch.Tensor([[0, 0.5], [2.1, 2.5]])

output = processor(video_frames=video_frames, frame_timestamps=frame_timestamps,
                   text=text, text_timestamps=text_timestamps)

n_ids = output['input_ids'].shape[0]
assert n_ids == output['timestamps'].shape[0]
assert n_ids == output['action_target_mask'].shape[0]
assert n_ids == output['text_target_mask'].shape[0]

print(output['input_ids'])
print(output['timestamps'])
print(output['action_target_mask'])
print(output['text_target_mask'])


print("All test cases passed!")