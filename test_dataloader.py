import h5py
import numpy as np
from tqdm import tqdm

def check_video_lengths(filepath):
    """
    Check whether all videos in the dataset have the same number of frames.
    
    Args:
        filepath (str): Path to the HDF5 file containing video data.
    
    Returns:
        bool: True if all videos have the same length, False otherwise.
    """
    with h5py.File(filepath, 'r') as f:
        video_ids = np.array(f['videoIDs'][:])
        unique_video_ids = np.unique(video_ids)
        
        video_lengths = []
        
        for video_id in tqdm(unique_video_ids, desc="Checking video lengths"):
            indices = np.where(video_ids == video_id)[0]
            video_lengths.append(len(indices))
        
        unique_lengths = np.unique(video_lengths)
        
        if len(unique_lengths) == 1:
            print(f"All videos have the same length: {unique_lengths[0]} frames.")
            return True
        else:
            print("Videos have different lengths:")
            for length in unique_lengths:
                print(f"{length} frames: {video_lengths.count(length)} videos")
            return False

filepath = "pikk_hackathon/frames_data.h5"  # Replace with the actual path to your dataset
check_video_lengths(filepath)
