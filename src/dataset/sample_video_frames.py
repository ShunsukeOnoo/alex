"""
#3 Sample video frames
This code samples video frames from videos in a directory and saves them as images.
The sampled images will be used for training a model to classify the video frames.
The total number of sampled images is determined by `frames_per_video` argument.

Args:
    video_dir (str): path to the directory containing videos
    output_dir (str): path to the directory to save the sampled images
    num_frames (int): (optional) number of frames in total to sample from videos
    sample_ratio (float): (optional) ratio of frames to sample from each video


選択肢
- 各動画から一定数のフレームをサンプリングする: 長い動画のフレームの割合が多くなる <- とりあえずこれを採用.
- 全動画の全フレームから一定数をサンプリングする: 短い動画だと類似したフレームが多くなる
"""
import os
import argparse
import concurrent.futures
import cv2
import numpy as np


def main(args):
    # list the all videos in the directory
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    video_paths = [os.path.join(args.video_dir, f) for f in video_files]

    # create saving directory
    os.makedirs(args.output_dir, exist_ok=True)

    # list of arguments for sample_frames function
    args_list = [
        video_paths,
        [args.output_dir]*len(video_paths),
        [args.frames_per_video]*len(video_paths)
    ]
    # sample frames using multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        for video_path, (success, error) in zip(video_files, executor.map(sample_frames, *args_list)):
            if success:
                print(f'Successfully sampled frames from {video_path}')
            else:
                print(f'Failed to sample frames from {video_path}', error)


def sample_frames(video_path, output_dir, n_samples):
    try:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frame indices to sample
        sample_indices = np.random.choice(np.arange(n_frames), size=int(n_samples), replace=False)
        sample_indices.sort()

        # sample frames
        sampled_frames = []  # tuple(image, frame_index)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # go to the frame
            ret, frame = cap.read()
            if ret:
                sampled_frames.append((frame, idx))
            else:
                # failed to read the frame
                return False, f'Failed to read frame {idx} from {video_path}'

        # save
        video_id = video_path.split('/')[-1].split('.')[0]
        for frame, idx in sampled_frames:
            cv2.imwrite(os.path.join(output_dir, f'{video_id}_{idx}.jpg'), frame)
    
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        return False, str(e)

    return True, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='path to the directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the directory to save the sampled images')
    parser.add_argument('--frames_per_video', type=int, required=True, help='number of frames to sample from each video')
    args = parser.parse_args()
    main(args)