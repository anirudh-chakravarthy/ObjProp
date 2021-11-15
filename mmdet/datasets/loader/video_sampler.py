from __future__ import division

import math
import torch
import numpy as np
import random

from torch.distributed import get_world_size, get_rank
from torch.utils.data.sampler import Sampler


class GroupVideoSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        self.vid_infos = dict()
        for i, size in enumerate(self.group_sizes):
            if size == 0: 
                continue
            indice = np.where(self.flag == i)[0]
                
            # store frame indices for each video
            vid_info = dict()
            for idx in indice:
                vid_id, _ = self.dataset.img_ids[idx]
                if vid_id not in vid_info.keys():
                    vid_info[vid_id] = list()
                vid_info[vid_id].append(idx)

            # count samples
            for (vid_id, frame_indice) in vid_info.items():
                self.num_samples += int(np.ceil(
                    len(frame_indice) / self.samples_per_gpu)) * self.samples_per_gpu
            self.vid_infos[i] = vid_info

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            
            for (vid_id, frame_indice) in self.vid_infos[i].items():
#                 frame_indice = frame_indice.copy()
#                 np.random.shuffle(frame_indice)
                num_extra = int(np.ceil(len(frame_indice) / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(frame_indice)
                frame_indice = np.concatenate([frame_indice, frame_indice[:num_extra]])
                indices.append(frame_indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = torch.from_numpy(indices).long()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples