import torch
import torch.nn as nn
from sympa.embeddings import ManifoldBuilder, Embeddings
from torch_geometric.nn import GCNConv, GATConv, SGConv, GINConv

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


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

class LPModelGNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.manifold = ManifoldBuilder.get(manifold=args.model, metric=args.metric, dims=args.dims)
        
        if args.gnn == 'gcn':
            self.conv1 = GCNConv(args.num_features, args.dims)
            self.conv2 = GCNConv(args.dims, args.dims)
            
        if args.gnn == 'gat':
            self.conv1 = GATConv(args.num_features, args.dims)
            self.conv2 = GATConv(args.dims, args.dims)
            
        if args.gnn == 'sgc':
            self.conv1 = SGConv(args.num_features, args.dims, K=1)
            self.conv2 = SGConv(args.dims, args.dims, K=1)
            
        if args.gnn == 'gin':
            self.conv1 = GINConv(nn=torch.nn.Linear(args.num_features, args.dims), eps = 0, train_eps = False)
            self.conv2 = GINConv(nn=torch.nn.Linear(args.dims, args.dims), eps = 0, train_eps = False)
    
    def encode(self, node_features, edge_index):
        
        node_features = self.conv1(node_features.float(), edge_index).relu()
        node_features = self.conv2(node_features, edge_index)
        
        return node_features

    def decode(self, node_features, edge_label_index):

        node_features = self.manifold.normalize(node_features)
      
        emb_in = node_features[edge_label_index[0]]
        emb_out = node_features[edge_label_index[1]]
        
        sqdist = self.similarity_score(emb_in, emb_out)
        probs = self.dc.forward(sqdist)
        
        return probs

    def similarity_score(self, lhs, rhs):
        dist = self.manifold.dist(lhs, rhs)
        return dist ** 2


class LPModelGNNProductEuclidean(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.metrics = args.metric.split(',')
        self.dims = args.dims // 2
        self.model = 'euclidean'
        
        # Assuming args.model and args.metric represent the two spaces.
        self.manifold1 = ManifoldBuilder.get(manifold=self.model, metric=self.metrics[0], dims=self.dims)
        self.manifold2 = ManifoldBuilder.get(manifold=self.model, metric=self.metrics[1], dims=self.dims)

        if args.gnn == 'gcn':
            self.conv1_1 = GCNConv(args.num_features, self.dims)  # First convolution for the first space
            self.conv2_1 = GCNConv(args.num_features, self.dims)  # First convolution for the second space
            
            self.conv1_2 = GCNConv(self.dims, self.dims)  # Second convolution for the first space
            self.conv2_2 = GCNConv(self.dims, self.dims)  # Second convolution for the second space
            
        elif args.gnn == 'gat':
            self.conv1_1 = GATConv(args.num_features, self.dims)
            self.conv2_1 = GATConv(args.num_features, self.dims)
            
            self.conv1_2 = GATConv(self.dims, self.dims)
            self.conv2_2 = GATConv(self.dims, self.dims)
            
        elif args.gnn == 'sgc':
            self.conv1_1 = SGConv(args.num_features, self.dims, K=1)
            self.conv2_1 = SGConv(args.num_features, self.dims, K=1)
            
            self.conv1_2 = SGConv(self.dims, self.dims, K=1)
            self.conv2_2 = SGConv(self.dims, self.dims, K=1)
            
        elif args.gnn == 'gin':
            self.conv1_1 = GINConv(nn=torch.nn.Linear(args.num_features, self.dims), eps=0, train_eps=False)
            self.conv2_1 = GINConv(nn=torch.nn.Linear(args.num_features, self.dims), eps=0, train_eps=False)
            
            self.conv1_2 = GINConv(nn=torch.nn.Linear(self.dims, self.dims), eps=0, train_eps=False)
            self.conv2_2 = GINConv(nn=torch.nn.Linear(self.dims, self.dims), eps=0, train_eps=False)
    
    def encode(self, node_features, edge_index):
        # Apply the first convolution for each space
        node_features1_1 = self.conv1_1(node_features.float(), edge_index).relu()
        node_features2_1 = self.conv2_1(node_features.float(), edge_index).relu()
        
        # Apply the second convolution for each space
        node_features1_2 = self.conv1_2(node_features1_1, edge_index)
        node_features2_2 = self.conv2_2(node_features2_1, edge_index)
        
        return torch.cat((node_features1_2, node_features2_2), dim=1)  # Concatenate features from both spaces
    
    def decode(self, node_features, edge_label_index):
        node_features1 = self.manifold1.normalize(node_features[:, :self.dims])
        node_features2 = self.manifold2.normalize(node_features[:, self.dims:])
      
        emb_in1 = node_features1[edge_label_index[0]]
        emb_out1 = node_features1[edge_label_index[1]]
        
        emb_in2 = node_features2[edge_label_index[0]]
        emb_out2 = node_features2[edge_label_index[1]]
        
        sqdist1 = self.similarity_score(emb_in1, emb_out1, self.manifold1)
        sqdist2 = self.similarity_score(emb_in2, emb_out2, self.manifold2)
        
        # Combine the similarity scores from both spaces
        sqdist = sqdist1 + sqdist2
        
        probs = self.dc.forward(sqdist)
        
        return probs

    def similarity_score(self, lhs, rhs, manifold):
        dist = manifold.dist(lhs, rhs)
        return dist ** 2
