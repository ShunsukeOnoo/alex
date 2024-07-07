"""
# 2 Download YouTube videos
Download YouTube videos using pytube and ProcessPoolExecutor.
Before running this script, you need to prepare a JSON file containing the YouTube video URLs.
You can use the whole MineDojo dataset file.

Usage:
    python download_youtube_videos.py --config path_to_json_file --output_dir path_to_output_dir
"""
import argparse
import json
import os
import sys
# from multiprocessing import Pool
import concurrent.futures
from pytube import YouTube


def load_json(path):
    """
    Returns:
        dataset: a list of dicts. Each dict contains at least the following keys:
            id, title, link
    """
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset


def download_and_save(data, output_dir):
    """
    Download the video and save to the output directory
    Filename format: {id}.mp4
    Return True if the video is successfully downloaded, False otherwise
    
    Args:
        data (dict): contains at least the following keys: id, title, link
        output_dir (str): path to the output directory
    Returns:
        success (bool): True if the video is successfully downloaded, False otherwise
        error (str): error message if the video is not successfully downloaded, None otherwise
    """
    yt = YouTube(data['link'])
    try:
        print('Start downloading video: ', data['id'])
        # ToDo: select the resolution flexibly
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().last()
        stream.download(output_path=output_dir, filename=f'{data["id"]}.mp4')
        print('Successfully downloaded video: ', data['id'])
        return True, None
    
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print('Failed to download video: ', data['id'], ' with error: ', e)
        return False, str(e)


def main(args):
    # load json file
    dataset = load_json(args.config)

    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # list of downloaded videos and failed videos
    downloaded_videos = []
    failed_videos = []

    # download the videos using multiprocessing
    # max_workers=None means the number of workers is equal to the number of processors on the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        for data, (success, e) in zip(dataset, executor.map(download_and_save, dataset, [args.output_dir]*len(dataset))):
            
            # if success is True, the video corresponding to data['id'] is successfully downloaded
            # else, the video is not successfully downloaded
            if success:
                downloaded_videos.append(data)
            else:
                data['error_msg'] = e
                failed_videos.append(data)

    # save the downloaded videos and failed videos to json files for recovery
    with open(os.path.join(args.output_dir, 'downloaded_videos.json'), 'w') as f:
        json.dump(downloaded_videos, f, indent=4)
    with open(os.path.join(args.output_dir, 'failed_videos.json'), 'w') as f:
        json.dump(failed_videos, f, indent=4)

    # print info
    print('Total number of videos: ', len(dataset))
    print('Successfully downloaded videos: ', len(downloaded_videos))
    print('Failed to download videos: ', len(failed_videos))
    print(f'Success rate: {len(downloaded_videos)/len(dataset):.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON file containing the YouTube video URLs')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    main(args)