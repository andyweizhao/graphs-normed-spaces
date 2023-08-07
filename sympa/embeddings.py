import abc
import torch
import torch.nn as nn
import geoopt as gt
from sympa.manifolds import Euclidean, EuclideanlMetricType, UpperHalf, BoundedDomain, ProductManifold
from geoopt.manifolds.siegel.vvd_metrics import SiegelMetricType

class Embeddings(nn.Module, abc.ABC):

    def __init__(self, num_embeddings, embedding_dim, manifold, _embeds):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        
        self.embeds = gt.ManifoldParameter(data=_embeds, manifold=self.manifold)

    def forward(self, input_index):

        return self.embeds[input_index]

    def proj_embeds(self):
        """Projects embeddings back into the manifold"""
        with torch.no_grad():
            self.embeds.data = self.manifold.projx(self.embeds.data)

    @abc.abstractmethod
    def norm(self):
        pass

    @classmethod
    def get(cls, model_name: str):
        if model_name in {"upper", "bounded", "dual", "spd"}:
            return MatrixEmbeddings
        if model_name in {"euclidean", "poincare", "lorentz", "sphere", "prod-hysph", "prod-hyhy", "prod-hyeu",
                          "prod-sphsph"}:
            return VectorEmbeddings
        raise ValueError(f"Unrecognized embedding model: {model_name}")


class MatrixEmbeddings(Embeddings):

    def __init__(self, num_embeddings, dims, manifold, init_eps=1e-3):
        
        _embeds = manifold.random(num_embeddings, dims, dims, from_=-init_eps, to=init_eps)
        super().__init__(num_embeddings, dims, manifold, _embeds)

    def norm(self):
        points = self.embeds.data
        points = points.reshape(len(points), -1)
        return points.norm(dim=-1)

class VectorEmbeddings(Embeddings):
    def __init__(self, num_embeddings, dims, manifold, init_eps=1e-3):
        _embeds = torch.Tensor(num_embeddings, dims).uniform_(-init_eps, init_eps)
        
        super().__init__(num_embeddings, dims, manifold, _embeds)
        self.proj_embeds()

    def norm(self):
        return self.embeds.data.norm(dim=-1)

def get_prod_hysph_manifold(dims):
    poincare = gt.PoincareBall()
    sphere = gt.Sphere()
    return ProductManifold((poincare, dims // 2), (sphere, dims // 2))

def get_prod_hyhy_manifold(dims):
    poincare = gt.PoincareBall()
    return ProductManifold((poincare, dims // 2), (poincare, dims // 2))

def get_prod_hyeu_manifold(dims, metric):
    poincare = gt.PoincareBall()
    metric = EuclideanlMetricType(metric)    
    euclidean = Euclidean(metric=metric)
    
    return ProductManifold((poincare, dims // 2), (euclidean, dims // 2))

class ManifoldBuilder:

    manifolds = {
        "poincare": lambda dims: gt.PoincareBall(),
        "lorentz": lambda dims: gt.Lorentz(),
        "sphere": lambda dims: gt.Sphere(),
        "prod-hysph": get_prod_hysph_manifold,
        "prod-hyhy": get_prod_hyhy_manifold,
        "spd": lambda dims: gt.SymmetricPositiveDefinite(),
    }

    manifolds_flexible_metrics = {
        "euclidean": lambda dims, metric: Euclidean(metric=metric),
        "upper": lambda dims, metric: UpperHalf(metric=metric),
        "bounded": lambda dims, metric: BoundedDomain(metric=metric),
        "prod-hyeu": lambda dims, metric: get_prod_hyeu_manifold(dims=dims, metric=metric),
    }

    @classmethod
    def get(cls, manifold, metric, dims):
        if manifold in cls.manifolds:
            return cls.manifolds[manifold](dims)
        
        if metric in EuclideanlMetricType._value2member_map_ :
            metric = EuclideanlMetricType(metric)

        if metric in SiegelMetricType._value2member_map_ :
            metric = SiegelMetricType(metric)
            
        manifold = cls.manifolds_flexible_metrics[manifold]
        return manifold(dims=dims, metric=metric)
    