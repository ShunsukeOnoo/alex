"""
#4 Classify video frames using OpenAI ChatGPT API
Classifies the sampled video frames into gameplay and non-gameplay frames using OpenAI ChatGPT API.

Args:
    frames_dir (str): path to the directory containing sampled video frames
    api_key (str, optional): OpenAI API key. If not provided, the script prompt to enter the API key.

ToDo:
- Continue the execution from the last frame to reduce cost.
- Arbitrary save path
- Use arbitrary text prompt
- Make it faster: currently ~5s per frame. there is a room for concurrent requests.
- Use other models
"""

import argparse
import os
import json
import base64
import requests
import time
from getpass import getpass
import tqdm

from util import openai_vision


# first prompt
PROMPT = '''\
Please help me identify screenshots that belong only to the gameplay of the Minecraft. \
Minor artifacts, such as a streamers face on a corner or window frame around the gameplay is fine. But if there are around 50% of artifacts, \
I consider it as a non-gameplay screenshot of Minecraft.\n\
Is this a gameplay screenshot of Minecraft? It it is, please only answer "Yes", otherwise only answer "No".\n\
'''

# prompt based on VPT (Baker et al., 2022)
# this prompt costs 239 tokens on the GPT-4 tokenizer
PROMPT_VPT = '''
Please help us identify screenshots that belong only to the survival mode in Minecraft. \
Everything else (Minecraft creative mode, other games, music videos, etc.) should be marked \
as None of the above. Survival mode is identified by the info at the bottom of the screen:
- a health bar (row of hearts)
- a hunger bar (row of chicken drumsticks)
- a bar showing items held

When answering, please only answer one of the following labels:
- Minecraft Survival Mode without Artifacts: These images will be clean screenshots\
from the Minecraft survival mode gameplay without any noticeable artifacts.
- Minecraft Survival Mode with Artifacts: These images will be valid survival\
mode screenshots, but with some added artifacts. Typical artifacts may include image\
overlays (a logo/brand), text annotations, a picture-in-picture of the player, etc.
- None of the Above: Use this category when the image is not a valid Minecraft survival screenshot. \
It may be a non-Minecraft frame or from a different game mode. In non-survival game modes such as \
the creative mode, the health/hunger bars will be missing from the image, the item hotbar \
may or may not be still present.\
'''


# default sleep time to avoid the API rate limit
# since the limit is 300K tokens per minute at the time of coding, 
# we are not worried about the rate limit
T_SLEEP = 0.5

# save filename
# SAVE_FILENAME = 'gameplay_classification_results.json'

def main(args):
    # load the API key
    if args.api_key is None:
        args.api_key = getpass('Enter the OpenAI API key: ')

    # list the all frames in the directory
    frame_files = [f for f in os.listdir(args.frames_dir) if f.endswith('.jpg')]
    if args.debug:
        frame_files = frame_files[:10]

    # If there is already a result file, load it and skip the frames that are already classified
    if os.path.exists(os.path.join(args.frames_dir, args.output_filename)):
        print('Found existing results file. Loading the results...')
        with open(os.path.join(args.frames_dir, args.output_filename), 'r') as f:
            results = json.load(f)
        frame_files = [f for f in frame_files if f not in results.keys()]
    else:
        results = {}

    # classify frames using OpenAI API
    for filename in tqdm.tqdm(frame_files):
        path = os.path.join(args.frames_dir, filename)

        try:
            response = openai_vision(PROMPT_VPT, path, args.api_key)
            
            if response.status_code != 200:
                # if the response is not successful, print the error message and continue
                print(f'Failed to classify {path} with error: {response.json()}')
                continue
            results[filename] = response.json()['choices'][0]['message']['content']

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(f'Failed to classify {path} with error: {e}')

        finally:
            # this block is executed regardless of whether the try block raises an exception
            # sleep for T_SLEEP seconds to avoid the API rate limit
            time.sleep(T_SLEEP)
    
    # save the answers to a json file
    with open(os.path.join(args.frames_dir, args.output_filename), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True, help='path to the directory containing sampled video frames')
    parser.add_argument('--output_filename', type=str, default='gameplay_classification_results.json', help='filename to save the classification results')
    parser.add_argument('--api_key', type=str, default=None, required=False, help='OpenAI API key. If not provided, the script prompt to enter the API key.')
    parser.add_argument('--debug', action='store_true', help='debug mode. only use the first 10 frames')
    args = parser.parse_args()

    main(args)

