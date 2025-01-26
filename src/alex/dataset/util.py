import base64
import requests

def openai_vision(text, image_path, api_key):
    """
    Call OpenAI API to get the generation on text + image.
    Based on the sample code from OpenAI API documentation:
        https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
        
    This function returns the response (requests.models.Response)
    You can access the generated text by:
        response.json()['choices'][0]['message']['content']

    Args:
        text (str): text prompt
        image_path (str): path to the image
        api_key (str): OpenAI API key
    Returns:
        response (requests.models.Response): response from the API
    """
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": text,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')