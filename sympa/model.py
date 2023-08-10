import torch
import torch.nn as nn
from sympa.embeddings import ManifoldBuilder, Embeddings

class DisModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.manifold = ManifoldBuilder.get(manifold=args.model, metric=args.metric, dims=args.dims)
        self.embeddings = Embeddings.get(args.model)(num_embeddings=args.num_points, dims=args.dims, manifold=self.manifold)
        
    def forward(self, input_triplet):

        src_index, dst_index = input_triplet[:, 0], input_triplet[:, 1]
        src_embeds = self.embeddings(src_index)                      
        dst_embeds = self.embeddings(dst_index)                     
        
        dist = self.manifold.dist(src_embeds, dst_embeds)   
        
        return dist

    def embeds_norm(self):
        return self.embeddings.norm()
    
class LPModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.manifold = ManifoldBuilder.get(manifold=args.model, metric=args.metric, dims=args.dims)
        self.embeddings = Embeddings.get(args.model)(num_embeddings=args.num_points, dims=args.dims, manifold=self.manifold)

    def distortion(self, input_triplet):

        src_index, dst_index = input_triplet[:, 0], input_triplet[:, 1]
        src_embeds = self.embeddings(src_index)                      
        dst_embeds = self.embeddings(dst_index)                     
        
        dist = self.manifold.dist(src_embeds, dst_embeds)   
        
        return dist
    
    def encode(self, node_features, edge_index):
        return self.embeddings.embeds

    def decode(self, node_features, edge_label_index):

        emb_in = node_features[edge_label_index[0]]
        emb_out = node_features[edge_label_index[1]]

        probs = (self.manifold.logmap0(emb_in) * self.manifold.logmap0(emb_out)).sum(-1)
        return probs

    def similarity_score(self, lhs, rhs):
        dist = self.manifold.dist(lhs, rhs)
        return dist ** 2
