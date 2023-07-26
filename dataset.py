from torch.utils.data import Dataset
from typing import Any

class MyDataeset(Dataset):
    def __init__(self, edges, t) -> None:
        """
        params: edges (num_samples, 2*d, 2)
        params: t (num_samples, 2*d+1)
        """
        super().__init__()
        self.edges = edges
        self.t = t
        assert edges.size(0) == t.size(0)
        self.len = self.edges.size(0)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index) -> Any:
        assert 0 <= index < self.len
        return self.edges[index], self.t[index]
    

class MyDataeset_3(Dataset):
    def __init__(self, p) -> None:
        """
        params: p (num_samples, 2*d+1, 2)
        """
        super().__init__()
        self.p = p
        self.len = self.p.size(0)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index) -> Any:
        assert 0 <= index < self.len
        return self.p[index]