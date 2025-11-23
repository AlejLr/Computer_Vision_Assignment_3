import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

class JesterDataset(Dataset):
    '''
    Assingment 3 gesture recognition dataset - Jester
    It assumes that the dataset and the labels are structured as:
        - there is a CSV file with lines formatted as:
            <video_id>;<label>
        - the label csv has one label per line
        - Each video path is a directory containing frames of the video
        
    This dataset is for the baseline, and returns only the center frame of each video.
    '''
    
    def __init__(self, data_root, csv_path, labels_csv_path, num_frames, temporal=False, transform = None, is_train=True):
        '''
        Arg:
            - data_root: root directory where the dataset is stored
            - csv_path: Path to the train/val CSV file
            - labels_csv_path: Path to labels CSV file.
            - transform: optional transform to be applied on a sample
            - num_frames: number of frames per sample (for the temporal model)
            - temporal: True if the dataset is used for the temporal model
            - is_train: True if the dataset is used for training
        '''
        self.num_frames = num_frames
        self.temporal = temporal
        self.is_train = is_train
        
        self.data_root = data_root
        self.csv_path = csv_path
        self.labels_csv_path = labels_csv_path
        self.transform = transform
        
        # Load labels
        self.labels_names = self._load_label_names(labels_csv_path)
        self.label_to_index = {name: idx for idx, name in enumerate(self.labels_names)}
        
        # Load samples
        self.samples = self._load_samples(csv_path)
        
        if len(self.samples) == 0:
            raise RuntimeError(f'No samples')
        
    def _load_label_names(self, labels_csv_path):
        label_names = []
        with open(labels_csv_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    label_names.append(name)
        return label_names
        
    def _load_samples(self, csv_path):
        samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                    
                try:
                    video_id, label_name = line.split(';', maxsplit=1)
                except ValueError:
                    raise ValueError(f"Cound not parse '{csv_path}': '{line}'")
                
                video_id = video_id.strip()
                label_name = label_name.strip()
                
                if label_name not in self.label_to_index:
                    raise ValueError(f"Unknown label '{label_name}' in file '{csv_path}'")
                
                label_index = self.label_to_index[label_name]
                samples.append((video_id, label_index))
                
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def _get_frame_paths(self, video_id):
            
        video_dir = os.path.join(self.data_root, video_id)
        if not os.path.isdir(video_dir):
            raise ValueError(f"Video directory '{video_dir}' does not exist")
            
        frame_files = [
            f for f in os.listdir(video_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(frame_files) == 0:
            raise RuntimeError(f"No image files found in directory '{video_dir}'")
            
        frame_files.sort()
        frame_paths = [os.path.join(video_dir, f) for f in frame_files]
        return frame_paths
        
    def _load_center_frame(self, video_id):
        frame_paths = self._get_frame_paths(video_id)
        center_index = len(frame_paths) // 2
        center_frame_path = frame_paths[center_index]
        img = Image.open(center_frame_path).convert('RGB')
        return img
    
    def _sample_frame_indices(self, num_total_frames):
        if self.num_frames <= 1:
            return [num_total_frames // 2]
        
        if self.num_frames > num_total_frames:
            return list(range(num_total_frames))[:self.num_frames]
        
        base_pos = np.linspace(0, num_total_frames - 1, self.num_frames)
        
        if not self.is_train:
            indices = np.round(base_pos).astype(int)
            return indices.tolist()
        
        jitter = np.random.uniform(-0.5, 0.5, size=self.num_frames)
        jittered_pos = base_pos + jitter
        jittered_pos = np.clip(jittered_pos, 0, num_total_frames - 1)
        indices = jittered_pos.astype(int)
        return indices.tolist()
    
    def _load_clip(self, video_id):
        
        frame_paths = self._get_frame_paths(video_id)
        num_total = len(frame_paths)
        indices = self._sample_frame_indices(num_total)
        
        images = []
        for idx in indices:
            frame_path = frame_paths[idx]
            img = Image.open(frame_path).convert('RGB')
            images.append(img)
            
        return images
            
        
    def __getitem__(self, index):
        
        video_id, label_index = self.samples[index]
        
        if self.temporal and self.num_frames > 1:
            images = self._load_clip(video_id)
            if self.transform is not None:
                
                images = [self.transform(img) for img in images]
                images =  torch.stack(images, dim=0)
            return images, label_index
            
        img = self._load_center_frame(video_id)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label_index