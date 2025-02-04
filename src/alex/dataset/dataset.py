"""
Dataset class for minecraft youtube videos.

Before using the classes in this file, you have to prepare the following files:
- dataset index file: a json file that contains the starting and ending timestamp for each clip
- predicted actions files: json files that contains the actions predicted by the IDM model
- video files: the downloaded youtube videos
- downloaded transcripts
"""
from typing import List
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class YouTubeDataset(Dataset):
    """
    Dataset class for youtube video clips.
    """
    def __init__(
            self, 
            dataset_index_path: str, 
            action_dir: str, 
            video_dir: str,
            transcripts_dir: str,
            video_suffix: str = '.mp4',
            transform = None
            ):
        """
        Args:
            dataset_index_path (str): path to the dataset index file
            action_dir (str): path to the directory containing the actions
            video_dir (str): path to the directory containing the video files
            transcripts_dir (str): path to the directory containing the transcripts
            video_suffix (str): the suffix of the video files. Default is '.mp4'
            transforms (callable) : A function that takes in a sample and returns a transformed version.
        """
        with open(dataset_index_path, 'r') as f:
            self.dataset_index = json.load(f)
        self.action_dir = action_dir
        self.video_dir = video_dir
        self.transcripts_dir = transcripts_dir
        self.video_suffix = video_suffix
        self.transform = transform

    def __len__(self):
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Obtain the data for a single video clip.

        By default, returns the following data:
            video_frames (torch.Tensor): RGB video frames in the clip.
                Shape (n_frames, 3, h, w), Dtype float32, Range [0, 1]
            frame_timestamps (torch.Tensor): timestamps for video frames in seconds.
                Shape (n_frames,)
            actions (List[dict]): a list of dictionaries containing the actions
            text (List[str]): a list of transcript strings
            text_timestamps (torch.Tensor): the start time and end time of each transcript.
        If transformes are given, apply the transforms to the sample here.
        """
        clip_info = self.dataset_index[idx]
        video_id = clip_info['video_id']
        start_idx, end_idx = clip_info['start_idx'], clip_info['end_idx']

        # obtain the video frames
        video_path = os.path.join(self.video_dir, video_id + self.video_suffix)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open the video file: {video_path}")
        frames, frame_timestamps = extract_video_frames(cap, start_idx, end_idx)
        cap.release()

        # obtain the predicted actions
        clip_id = clip_info['clip_id']
        action_path = os.path.join(self.action_dir, f'{video_id}_{clip_id}.json')
        with open(action_path, 'r') as f:
            actions = json.load(f)

        # obtain the transcripts
        transcript_path = os.path.join(self.transcripts_dir, video_id + '.json')
        with open(transcript_path, 'r') as f:
            transcripts = json.load(f)
        text, text_timestamps = extract_transcripts(
            transcripts, clip_info['start_timestamp'], clip_info['end_timestamp']
        )

        data = {
            'video_frames': frames,
            'frame_timestamps': frame_timestamps,
            'actions': actions,
            'text': text,
            'text_timestamps': text_timestamps,
        }

        if self.transform:
            data = self.transform(data)
        return data


def extract_video_frames(cap: cv2.VideoCapture, start_idx: int, end_idx: int):
    """
    Extract video frames that are within the given range.

    Returns:
        video_frames (torch.Tensor]): RGB video frames. Shape (n_frames, 3, h, w), Dtype float32, Range [0, 1] 
        timestamps (torch.Tensor): timestamps for video frames in seconds. Shape: (n_frames,)
    """
    video_frames = []
    timestamps = []   # absolute timestamps in seconds

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    n_frames = int(end_idx - start_idx + 1)
    for _ in range(n_frames):
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        ret, frame = cap.read()
        if not ret:
            # TODO: warning or error message
            break
        video_frames.append(frame)
    
    # turn video frames into a tensor
    video_frames = np.stack(video_frames)[..., ::-1]  # BGR -> RGB
    video_frames = video_frames.transpose(0, 3, 1, 2) # (n_frames, h, w, 3) -> (n_frames, 3, h, w)

    # TODO: hard coding the dtype may not be a good idea
    # .copy() is to avoid ValueError: At least one stride in the given numpy array is negative,
    video_frames = torch.tensor(video_frames.copy(), dtype=torch.float32) / 255.0

    # transform the absolute timestamps to relative timestamps
    timestamps = np.array(timestamps)
    timestamps = timestamps - timestamps[0]
    return video_frames, timestamps


def extract_transcripts(transcript_file: List[dict], start_time: float, end_time: float) -> List[str]:
    """
    Extract the transcripts within the given time range.

    Args:
        transcripts (List[dict]): a list of dictionaries containing the transcript information
        start_time (float): the starting time of the clip
        end_time (float): the ending time of the clip

    Returns:
        transcripts (List[str]): a list of transcript strings
        timestamps (torch.Tensor): the start time and end time of each transcript.
            Shape (n_transcripts, 2)
    """
    # extract the transcripts within the given time range
    extracted = [t for t in transcript_file if t['start'] >= start_time and t['start'] + t['duration'] <= end_time]

    transcripts = [t['text'] for t in extracted]
    timestamps = [(t['start'], t['start'] + t['duration']) for t in extracted]

    # turn the timestamps into a tensor and into a relative timestamp
    timestamps = torch.tensor(timestamps)
    timestamps = timestamps - start_time
    return transcripts, timestamps