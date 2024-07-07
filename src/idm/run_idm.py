"""
Run inverse dymnamics model (IDM) on video dataset to obtain actions.

Preparation for running this script:
- Clone VPT repo into lib/vpt (You have to change the name of the repo to vpt)
- Download IDM model parameters (.model and .weight files) from the link in the VPT repo.
- Prepare the dataset using the scripts in data_util directory of my repo.
- Build and activate the environment for IDM following the instructions in the VPT repo.

Output format:
- For each video clip, this script will output a json file that contains the actions for each frame.

TODO:
- 複数のGPUとmultiprocessingを使う
- Add fancy progress bar
"""
from argparse import ArgumentParser
import os
import pickle
import cv2
import numpy as np
import json
import sys
sys.path.append('./lib/vpt/')
from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent


def load_model(model: str, weights: str, device: str = 'cuda'):
    """
    Load the IDM model from the given model and weights files.
    Based on the code in Video-PreTraining/run_inverse_dynamics_model.py

    Args:
        model: Path to the '.model' file to be loaded.
        weights: Path to the '.weights' file to be loaded.
    Returns
        agent: IDM agent loaded with the model and weights
    """
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=device)
    agent.load_weights(weights)
    return agent


def stack_actions(actions: list):
    """
    Stack the actions for a video clip

    Args:
        actions: Predicted actions for each batch. 
            Each action is a dict that contains 2d numpy array for each key.
    Returns:
        stacked_actions (dict): Stacked actions for the video clip. Each value is a 1d list.
    """
    keys = list(actions[0].keys())
    stacked_actions = {
        key: np.squeeze(np.concatenate([action[key] for action in actions], axis=1), axis=0).tolist()
        for key in keys
    }
    return stacked_actions


def main(model, weights, dataset_json, video_dir, output_dir, batch_size, device):
    os.makedirs(output_dir, exist_ok=True)
    # Load the dataset index file that contains the vido frames to use for IDM predictions
    with open(dataset_json) as json_file:
        dataset = json.load(json_file)
    print('Number of video clips', len(dataset))

    agent = load_model(model, weights, device)
    required_resolution = ENV_KWARGS["resolution"]

    # For each video, load the frames, run IDM, and save the results
    # clip: dict that contains the information of the video clip
    for clip in dataset:
        print('Start processing', clip['video_id'], clip['clip_id'])

        # 1. Load the video frames
        video_path = os.path.join(video_dir, f'{clip["video_id"]}.mp4')
        cap = cv2.VideoCapture(video_path)
        # TODO: rather than setting fps, assert the fps is correct
        cap.set(cv2.CAP_PROP_FPS, clip['fps'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip['start_idx'])

        # Load the frames to predict the actions
        n_frames = int(clip['end_idx'] - clip['start_idx'] + 1) # we can use the last frame
        frames = []
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f'Failed to read the frame from {clip["video_id"]}')
            
            # check the resolution of the frame
            assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
            frames.append(frame)
        cap.release()

        # 2. Run IDM on the frames
        n_frames = len(frames)
        n_batches = int(np.ceil(n_frames / batch_size))
        predicted_actions = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, n_frames)
            batch_frames = np.stack(frames[start:end])
            batch_frames = batch_frames[..., ::-1]  # BGR -> RGB
            
            batch_actions = agent.predict_actions(batch_frames)  # Dict of actions
            predicted_actions.append(batch_actions)

        # 3. Save the results
        predicted_actions = stack_actions(predicted_actions)
        output_file = os.path.join(output_dir, f'{clip["video_id"]}_{clip["clip_id"]}.json')
        with open(output_file, 'w') as f:
            json.dump(predicted_actions, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # arguments for IDM model
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")

    # arguments for video dataset
    parser.add_argument('--dataset-json', type=str, required=True, help='Path to the dataset json file.')
    parser.add_argument('--video-dir', type=str, required=True, help='Path to the directory containing the videos.')

    # arguments for saving the actions
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the directory to save the actions.')

    # arguments for running IDM
    parser.add_argument('--batch-size', type=int, default=16, help='Number of frames to process at a time.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the IDM model.')

    args = parser.parse_args()
    main(**vars(args))

