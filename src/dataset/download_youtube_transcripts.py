"""
Download youtube transcripts using youtube_transcript_api and ProcessPoolExecutor
ToDo: Do we need concurrency?

Usage:
    python download_youtube_transcripts.py --config path_to_json_file --output_dir path_to_output_dir
"""
import argparse
import json
import os
import sys
import concurrent.futures
from youtube_transcript_api import YouTubeTranscriptApi as YT
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


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
    Download the transcripts and save to the output directory
    Filename format: {id}.json
    Return True if the video is successfully downloaded, False otherwise
    
    Args:
        data (dict): contains at least the following keys: id, title, link
        output_dir (str): path to the output directory
    Returns:
        success (bool): True if the video is successfully downloaded, False otherwise
        error (str): error message if the video is not successfully downloaded, None otherwise
    """
    id = data['id']
    print('Start downloading transcript: ', id)

    try:
        # first try to get the English transcript
        transcript = YT.get_transcript(id)

    except NoTranscriptFound as e:
        # there may be no English transcript, try to get the original transcript
        try:
            transcript_original = list(YT.list_transcripts(id))[0]
            transcript = transcript_original.translate('en').fetch()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print('Failed to download transcript: ', id, ' with error: ', e)
            return False, str(e)
        
    except TranscriptsDisabled as e:
        # probably the video itself is not available
        print('Failed to download transcript: ', id, ' with error: ', e)
        return False, str(e)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print('Failed to download transcript: ', id, ' with error: ', e)
        return False, str(e)
    
    # arriving here means the transcript is successfully downloaded
    with open(os.path.join(output_dir, f'{id}.json'), 'w') as f:
        json.dump(transcript, f, indent=4)
    print('Successfully downloaded transcript: ', id)
    return True, None


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
    info = {
        'downloaded_videos': downloaded_videos,
        'failed_videos': failed_videos,
    }
    with open(os.path.join(args.output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

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