import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale, ApplyTransformToKey
from pytorchvideo.data.encoded_video import EncodedVideo


class PackPathway(torch.nn.Module):
    """
    Converts a single video tensor into SlowFast pathways (slow and fast tensors).
    Slow pathway is downsampled by a factor of `alpha`.
    """

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        if frames.size(1) == 0:
            raise ValueError("Input tensor has zero frames. Check your preprocessing or input video.")

        # Fast Pathway: Full frame rate
        fast_pathway = frames

        # Slow Pathway: Downsampled by `alpha`
        slow_pathway_indices = torch.linspace(
            0, frames.shape[1] - 1, max(1, frames.shape[1] // self.alpha)
        ).long()
        slow_pathway = torch.index_select(frames, 1, slow_pathway_indices.to(frames.device))

        return [slow_pathway, fast_pathway]


class TestVideoDataset(Dataset):
    """
    A dataset class for processing video files without predefined labels, intended for testing and inference.

    Attributes:
        root_dir (str): Path to the directory containing video files.
        frame_size (tuple): Target frame dimensions (height, width).
        num_frames (int): Number of frames to sample from each video.
        transform (torchvision.transforms.Compose): Preprocessing pipeline for the video data.
    """

    def __init__(self, root_dir, frame_size, num_frames, transform=None):
        self.root_dir = root_dir
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.samples = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith('.mp4')]

        # Default preprocessing pipeline
        def normalize(x):
            return x / 255.0

        if transform is None:
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames),  # Sample fixed number of frames
                        normalize,  # Normalize pixel values
                        ShortSideScale(size=self.frame_size[0]),  # Resize shorter side
                        CenterCropVideo(self.frame_size[0]),  # Center crop to target size
                        PackPathway()  # Prepare SlowFast pathways
                    ]
                )
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        video = EncodedVideo.from_path(video_path)

        # Extract frames from the full video duration
        start_time = 0
        clip_duration = int(video.duration)
        end_sec = start_time + clip_duration

        # Preprocess video frames
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data_tensor = self.transform(video_data)
        inputs = video_data_tensor["video"]

        return inputs, video_path
