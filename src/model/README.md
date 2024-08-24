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
- `input_ids` (torch.Tensor): Token ids that contians the video frame placeholders.
    Shape (n_tokens,). Placeholder ids are the same for all the video frames. Each video frame may corresponds to multiple placeholders when frame embeddings are more than one token.
- `video_frames` (torch.Tensor): Preprocessed video frames. Shape (n_frames, num_channel, height, width).
- `timestamp` (torch.Tensor): Timestamp for each tokens. Shape (n_tokens,).
- `video_frame_mask` (torch.Tensor): Indicates a position in input_ids that corresponds to a video frame. Shape (n_tokens,). 0 for text tokens and 1 for video frames.
- `actin_target_mask` (torch.Tensor): Indicates the position that predicts the action. Shape (n_tokens,). 1 for the position that predicts the action. 0 for the others. For each video frame, the lsat embedding position will be 1.
- `text_target_mask` (torch.Tensor): Indicates the position that predicts the text. Shape (n_tokens,). 1 for the position that predicts the text. 0 for the others. Positions whose next token is a text token will be 1.
- `actions` (torch.Tensor): (Optional) Preprocessed actions that corresponds to each video frames. Shape (n_frames, action_dim).

TODO
- Inputs may be only text or video frames on inference.
- Inputs may lack timestamps on inference.
  - If inputs are text + video, then text then video.
  - If inputs are only video frames, use default fps to create pseudo timestamps.
- The `action_target_mask` and `text_target_masks` can be bool tensors rather than integer tensors.


### Unresolved
Where is the best time to apply the preprocessing?
- At dataset: Use `transform` parameter of the dataset. It's simple but may not useful when we want to apply padding.
- At dataloader: 
- At batch




## AlexModel
`modeling_alex.py` has two model classes.
- `AlexTextModel`: The base model that wraps the language model, performs the merging of vision and text embeddings, and add time embeddings.
- `AlexForAction`: Model with `action_head`, `image_encoder`, and `image_projection`.

Both models are the subclasses of `PreTrainedModel`, so that:
- You can use `AlexModel.from_pretrained(ckpt_dir_or_path)` to load model weights,
- You can use other methods it offers.


### Time embedding
There can be several methods to embed time information into the input. The difference to the normal language model is that the time is continuous but the position is discrete. To effectively utilize the time information, there are several options for embedding time. Right now, I have implemented the following methods:
- Time2Vec (Kazemi et al., 2019).

If you think it's okay to ignore the timestamps, you can just use sinusoidal positional encoding, just like the ordinary language models.

### Config
```yaml
text_config:
  timestamp_encoding: time2vec

vision_config:

vision_projection_config:

hidden_size:
action_dim:
```


### `AlexTextModel`


### `AlexForAction`