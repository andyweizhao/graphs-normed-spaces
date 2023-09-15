import torch
import torch.nn as nn
from sympa.embeddings import ManifoldBuilder, Embeddings

class RecoSys(nn.Module):
    """Recommender system model that operates on different Manifolds"""
    def __init__(self, args):
        super().__init__()
        
        self.manifold = ManifoldBuilder.get(manifold=args.model, metric=args.metric, dims=args.dims)
        self.embeddings = Embeddings.get(args.model)(num_embeddings=args.num_points, dims=args.dims, manifold=self.manifold)
        
        self.bias_lhs = torch.nn.Parameter(torch.zeros(args.num_points), requires_grad=False)
        self.bias_rhs = torch.nn.Parameter(torch.zeros(args.num_points), requires_grad=False)

    def forward(self, input_triplet):
        """
        Calculates and returns the score for a pair (head, tail), based on the distances in the space.

        :param input_triplet: tensor with indexes of embeddings to process. (b, 2)
        :return: scores: b
        """
        lhs_index, rhs_index = input_triplet[:, 0], input_triplet[:, -1]
        lhs_embeds = self.embeddings(lhs_index)                       # b x 2 x n x n or b x n
        rhs_embeds = self.embeddings(rhs_index)                       # b x 2 x n x n
        lhs_bias = self.bias_lhs[lhs_index]                           # b
        rhs_bias = self.bias_rhs[rhs_index]                           # b

        sq_distances = self.distance(lhs_embeds, rhs_embeds) ** 2
        
        scores = lhs_bias + rhs_bias - sq_distances
        
        return scores

    def distance(self, src_embeds, dst_embeds):
        """
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In complex matrix spaces, it will be of the shape b x 2 x n x n. In Vector spaces it will be b x n
        :return: tensor of b with distances from each src to each dst
        """
        return self.manifold.dist(src_embeds, dst_embeds).reshape(-1)   # b x 1
