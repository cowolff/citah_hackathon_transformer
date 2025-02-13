import h5py
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

# function to get frames of video:
def get_frames_of_certain_video(reader, videoid):
    """
    Retrieve frames of a specific video from an HDF5 file.

    This function loads video frames and their corresponding video IDs from 
    an HDF5 file ('frames_data.h5') and extracts frames that belong to the given video ID.

    Args:
        videoid (int): The ID of the video whose frames are to be retrieved.

    Returns:
        np.ndarray: A NumPy array containing frames of the specified video. 
    
    Example:
        frames = get_frames_of_certain_video(0)
        print(frames.shape)  # Expected output: (100, 32, 64)
    """
        
    frames_np = np.array(reader['frames'][:])
    floorline_np = np.array(reader['floorline_masks'][:])
    labels_np = np.array(reader['labels'][:])

    video_ids = reader['videoIDs'][:]
    videoids_np = np.array(video_ids)
    
    indices_of_frames = np.where(videoids_np == videoid)
    
    indices = indices_of_frames[0]

    frames_of_video = frames_np[indices]
    # Normalize the pixel values to the range [0, 1]
    frames_of_video = frames_of_video / 255.0
    # Resize to 32x32
    frames_of_video = np.array([resize(frame, (32, 32), anti_aliasing=True) for frame in frames_of_video])
    
    floorline_of_video = floorline_np[indices]

    floorline_of_video = np.array([resize(floorline, (32, 32), anti_aliasing=True) for floorline in floorline_of_video])

    labels_of_video = labels_np[indices]


    if len(np.unique(labels_of_video)) != 1:
        raise ValueError("All labels should be the same.")
    else:
        label = labels_of_video[0]
    
    return frames_of_video, floorline_of_video, label

class VideoDataLoader(Dataset):

    def __init__(self, filepath):
        super().__init__()

        # get all video ids
        with h5py.File(filepath, 'r') as f:
            video_ids = f['videoIDs'][:]
            video_ids = np.array(video_ids)
            # get unique video ids
            unique_video_ids = np.unique(video_ids)

            # Filter out video ids with length < 100
            valid_video_ids = []
            for video_id in tqdm(unique_video_ids, desc="Filtering video IDs"):
                indices_of_frames = np.where(video_ids == video_id)[0]
                if len(indices_of_frames) >= 100:
                    valid_video_ids.append(video_id)

            self.video_ids = np.array(valid_video_ids)

        self.reader = h5py.File(filepath, 'r')
        self.filepath = filepath

        # Calculate the share of each label
        self.label_counts = {0: 0, 1: 0}
        labels = self.reader['labels'][:]
        for label in labels:
            self.label_counts[label[0]] += 1

        total_videos = len(labels)
        self.label_share = {label: count / total_videos for label, count in self.label_counts.items()}

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames, floorline, label = get_frames_of_certain_video(self.reader, video_id)
        frames = frames[:100]
        floorline = floorline[:100]
        return frames, floorline, label[0]