# Preprocessing and Models
Preprocessing data and modleing!

Files & Directories
- modeling_alex.py
- processing_alex.py

## Preprocessing
**TODO**: Fix this part


The processor `AlexProcessor` receives a sample or a batch from `YouTubeDataset` and performes preprocessing on them. The processor handles the following processes:
1. Tokenize the input text,
2. Interleave text tokens and video placeholders.
3. Preprocess the video frames.
4. Preprocess the actions.

The processor returns a `dict` with following keys and values:
- input_ids: Tokenized text with placeholders for video frames.
- video_frames: Preprocessed video frames with shape (batch, n_frames, height, wideth).
- timestamps: Timestamps for `input_ids` with shape (batch, n_tokens)
- video_frame_mask: A tensor that shows the positions of video frames in `input_ids`. Shape (batch, n_tokens,). 1 stands for video frames and 0 stands for text.
- actions: A tensor that shows actions with shape (batch, n_frames, action_dim).

TODO
- Inputs may be only text or video frames on inference.
- Inputs may lack timestamps on inference.
  - If inputs are text + video, then text then video.
  - If inputs are only video frames, use default fps to create pseudo timestamps.



## AlexModel
`modeling_alex.py` has two model classes.
- `AlexModel`: Base model without a head.
- `AlexModelForConditionalGeneration`: Model with `action_head`.

Both models are the subclasses of `PreTrainedModel`, so that:
- You can use `AlexModel.from_pretrained(ckpt_dir_or_path)` to load model weights,
- You can use other methods it offers.

### AlexModel.forward()

The forward pass of the model. It will:
1. Turn `input_ids` to embeddings.
2. Encode `video_frames` using the vision encoder.
3. Combine both embeddings using `video_frame_mask`.
4. Create the temporal encoding using timestamps and add it to the embedding.
5. Input to the base Transformer model.