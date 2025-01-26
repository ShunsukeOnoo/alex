"""
Filter video frames using the filtering model. The results will be saved in a json file
in specified directory for each video.
The resulting json files will contain the following information:
- frame_indices: List of indices of frames in the result.
- results: List of predicted labels for each frame.

Features
- Use multiple GPUs and multiprocessing for parallel processing.

For detail, see the README.md in the same directory.
"""
import argparse
import os
import sys
import json
import joblib
import concurrent.futures

import tqdm
import numpy as np
import sklearn
import torch
import clip
import cv2
from PIL import Image

# for cuda compatibility with multiprocessing
import torch.multiprocessing as multiprocessing
if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)
    print("{} setup done".format(multiprocessing.get_start_method()))



def load_clf_model(clip_model_name, clf_model_path, device):
    """
    Load the classifier model. The classifier model consists of
    the CLIP model and the scikit-learn model.

    Args:
        clip_model_name (str): Name of the CLIP model.
        clf_model_path (str): Path to the scikit-learn model.
        device (str): Device to use.
    Returns:
        clip_model (torch.nn.Module): CLIP model.
        preprocess (torch.nn.Module): Preprocessing module for the CLIP model.
        clf_model (sklearn.base.BaseEstimator): Scikit-learn model.
    """
    print('Loading classifier model on device: ', device)
    # load the CLIP model
    clip_model, preprocess = clip.load(clip_model_name, device=device)

    # load the scikit-learn model
    clf_model = joblib.load(clf_model_path)

    print('Classifier model loaded on device: ', device)
    return clip_model, preprocess, clf_model


def classify_frames(clf_models, video_dir, video_id, sampling_fps, output_dir, batch_size, device_index):
    """
    Classify frames in a given video and save the results.
    Use the CLIP model unique to each GPU.

    Args:
        clf_models (list): List of CLIP models, preprocessing modules, and scikit-learn models.
        video_path (str): Path to the video.
        sampling_fps (float): Sampling fps.
        output_dir (str): Path to the output directory.
        batch_size (int): Batch size.
        device_index (int): Index of the GPU to use.
    """
    print('Start processing video: ', video_id, ' on device: ', device_index)
    if not os.path.exists(os.path.join(video_dir, video_id + ".mp4")):
        return False, "Video does not exist"
    device = f"cuda:{device_index}"
    try:
        # model to use
        clip_model, preprocess, clf_model = clf_models[device_index]

        # load the video
        cap = cv2.VideoCapture(os.path.join(video_dir, video_id + ".mp4"))

        # Make batches for this video
        # this is suboptimal because the bach may be smaller than 
        # the batch_size at the end of the video
        batchgen = BatchGenerator(cap, sampling_fps, batch_size)

        # classify the frames in the video
        processed_frames = []
        results = []
        for frame_indices, batch in batchgen:
            inputs = torch.stack([preprocess(frame) for frame in batch]).to(device)
            with torch.no_grad():
                features = clip_model.encode_image(inputs).cpu().numpy()
            pred = clf_model.predict(features)  # np.ndarray containing the predicted labels

            processed_frames += frame_indices
            results += pred.tolist()
            
        # save the results
        results = {
            'video_id': video_id,
            'frame_indices': processed_frames,
            'results': results,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'sampling_fps': sampling_fps,
        }
        with open(os.path.join(output_dir, video_id + ".json"), "w") as f:
            json.dump(results, f)

        print('Finished processing video: ', video_id, ' on device: ', device_index)
        return True, None
    
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print("Failed to classify frames in video: ", video_id, " with error: ", e)
        return False, str(e)


class BatchGenerator:
    """
    Generate batches from video frames.
    """
    def __init__(self, cap, sampling_fps, batch_size):
        """
        Args:
            cap (cv2.VideoCapture): VideoCapture object.
            sampling_fps (float): Sampling fps.
            batch_size (int): Batch size.
        """
        self.cap = cap
        self.sampling_fps = sampling_fps
        self.batch_size = batch_size
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.skip_frames = int(self.fps / sampling_fps) - 1
        self.total_samples = int(self.total_frames / (self.skip_frames + 1))
        self.total_batches = int(np.ceil(self.total_samples / batch_size)) 

        self._i = 0

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Returns:
            frame_indices (list): List of indices of frames in the batch.
            batch (list): List of frames (PIL.Image)
        """
        if self._i > self.total_batches:
            raise StopIteration()
        
        frame_indices = []  # indices of frames to be sampled
        batch = []
        for _ in range(self.batch_size):
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if not ret:
                break

            # convert the frame into RGB Image
            frame = frame[:, :, ::-1]
            frame_indices.append(current_frame)
            batch.append(Image.fromarray(frame))

            # skip frames to the next sample if there are more frames
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            next_frame = current_frame + self.skip_frames
            if next_frame < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
            else:
                break

        self._i += 1

        if len(batch) == 0:
            raise StopIteration()

        return frame_indices, batch

        
def main(args):
    # make result dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load list of the videos
    with open(args.index_path, "r") as f:
        videos = json.load(f)  # list containing dicts with video info
    video_ids = [video["id"] for video in videos]

    # check existing videos in the directory
    video_ids = [video_id for video_id in video_ids if os.path.exists(os.path.join(args.video_dir, video_id + ".mp4"))]
    print('Found {} videos'.format(len(video_ids)))

    # check existing results
    if not args.clobber:
        print('Clobber is off. Checking existing results.')
        result_ids = [os.path.splitext(f)[0] for f in os.listdir(args.output_dir) if f.endswith('.json')]
        video_ids = [video_id for video_id in video_ids if video_id not in result_ids]
        print('Found {} videos without results'.format(len(video_ids)))

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise ValueError("No GPU available")

    # load the model for each GPU
    print('Loading classifiter models')
    clf_models = []
    for i in range(n_gpus):
        models = load_clf_model(args.clip_model_name,args.clf_model_path,f"cuda:{i}")
        clf_models.append(models)

    # classify frames using multiprocessing
    # each process will use a GPU
    # assign videos to processes, so that no video is processed by two processes
    # ues classify_frames
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_gpus) as executor:
        
        futures = [
            executor.submit(
                classify_frames,
                clf_models,
                args.video_dir,
                video_id,
                args.sampling_fps,
                args.output_dir,
                args.batch_size,
                i % n_gpus,
            )
            for i, video_id in enumerate(video_ids)
        ]

        # collect results
        for future in concurrent.futures.as_completed(futures):
            finished, e = future.result()
            if finished:
                #print("Finished processing video")
                pass
            else:
                print("Failed to process video with error: ", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True, help="Path to the index file")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing the videos")
    parser.add_argument("--clip_model_name", type=str, required=True, help="Name of the CLIP model")
    parser.add_argument("--clf_model_path", type=str, required=True, help="Path to the scikit-learn model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--sampling_fps", type=float, default=1, help="Sampling fps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing result files')
    args = parser.parse_args()
    main(args)