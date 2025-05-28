import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable, Iterator, List, Tuple

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self. shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn else self.default_collate
        self._indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator:
        if self.shuffle:
            random.shuffle(self._indices)
        self._current_idx = 0
        return self
    
    def __next__(self):
        if self._current_idx >= len(self._indices):
            raise StopIteration
        batch_indices = self._indices[self._current_idx:self._current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self._current_idx += self.batch_size
        return self.collate_fn(batch)
    
    @staticmethod
    def default_collate(batch: List[Tuple[np.ndarray, np.ndarray]]):
        data = np.stack([item[0] for item in batch])
        labels = np.array([item[1] for item in batch])
        return data, labels
