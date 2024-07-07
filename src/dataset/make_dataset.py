import argparse
import os
import json
import tqdm


def make_clip_info(video_id, start_idx, end_idx, fps, clip_count):
    """
    Make a dictionary containing information about a video clip.
    """
    clip_info = {
        "video_id": video_id,
        "clip_id": clip_count,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_timestamp": start_idx / fps,
        "end_timestamp": end_idx / fps,
        "fps": fps
    }
    return clip_info


def extract_clips(filter_results, min_duration, max_duration):
    """
    Extract video clips to be used for training, based on the fitering results.

    Args:
        filter_results (dict): Filtering results of the video.
            It should contain the following keys:
            - video_id
            - frame_indices
            - results
            - fps
            - sampling_fps
        min_duration (int): Minimum duration of the video in seconds.
        max_duration (int): Maximum duration of the video in seconds.
    Returns:
        clips (list): List of tuples (video_id, start_timestamp, end_timestamp)
    """
    video_id = filter_results['video_id']
    frame_indices = filter_results['frame_indices']
    frame_classes = filter_results['results']
    video_fps = filter_results['fps']

    clips = []
    clip_count = 0  # count the number of clips
    start_idx = None
    for idx, cls in zip(frame_indices, frame_classes):
        if cls in [0, 1]:  
            # this frame is okay
            if start_idx is None:
                # this is the first frame of the clip
                start_idx = idx

            if idx - start_idx >= max_duration * video_fps:
                # this clip reached max_duration: close the current clip
                clips.append(make_clip_info(video_id, start_idx, idx, video_fps, clip_count))
                clip_count += 1
                start_idx = None

        else:  
            # this frame is not okay
            if start_idx is not None and idx - start_idx >= min_duration * video_fps:
                # the clip is long enough: close the current clip
                # the current frame can not be used: idx-1
                clips.append(make_clip_info(video_id, start_idx, idx-1, video_fps, clip_count))
                clip_count += 1

            # reset the start index anyway
            start_idx = None

    return clips


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # files that save the filtering results
    filter_result_files = [f for f in os.listdir(args.filter_result_dir) if f.endswith('.json')]

    dataset = []  # list of tuples (video_id, start_timestamp, end_timestamp)
    for filter_result_file in tqdm.tqdm(filter_result_files):
        with open(os.path.join(args.filter_result_dir, filter_result_file), "r") as f:
            filter_results = json.load(f)

        clips = extract_clips(filter_results, args.min_duration, args.max_duration)
        dataset += clips

    with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_result_dir', type=str, required=True, help='path to the directory containing the filter results')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--min_duration', type=int, default=4, help='minimum duration of the video in seconds')
    parser.add_argument('--max_duration', type=int, default=32, help='maximum duration of the video in seconds')
    args = parser.parse_args()
    main(args)