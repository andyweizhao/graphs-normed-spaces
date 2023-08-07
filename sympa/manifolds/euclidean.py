import torch
import geoopt as gt
from enum import Enum
    
class EuclideanlMetricType(Enum):    
    L1 = 'l1'
    L2 = 'l2'
    LINFINITY = 'linf'
    
def compute_L1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return torch.sum(torch.abs(x - y), dim=-1)

def compute_L2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return torch.norm(x - y, dim=-1)

def compute_Linf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return torch.max(torch.abs(x - y), dim=-1)[0]
        
class Euclidean(gt.Euclidean):

    def __init__(self, ndim=1, metric=EuclideanlMetricType.L2):
        super().__init__(ndim=ndim)
        
        self.ndim = ndim
        if metric == EuclideanlMetricType.L1:
            self.compute_metric = compute_L1
            
        elif metric == EuclideanlMetricType.L2:
            self.compute_metric = compute_L2
            
        elif metric == EuclideanlMetricType.LINFINITY:
            self.compute_metric = compute_Linf
            
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    def dist(self, z1: torch.Tensor, z2: torch.Tensor):
        
        return self.compute_metric(z1, z2)

    def dist2(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False, dim=-1):
        
        return self.compute_metric(z1, z2) ** 2
    
    def logmap0(self, x: torch.Tensor, *, dim=-1):
        
        return x