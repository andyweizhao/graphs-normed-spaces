import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DatasetLoader
from sympa.model import DisModel
from sympa.eval.metrics import AverageDistortionMetric
import torch
import argparse
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
})
    
def plot_graph_edges(src_dst_ids, graph_dists, norm_dists, graph):
    
    vmin = 0
    vmax = 1
    
    for i in range(len(src_dst_ids)):
        gdist = graph_dists[i]
        ndist = norm_dists[i]
        if gdist != 1: continue 

        src, dst = src_dst_ids[i].tolist()        
        
        distortion = metric.calculate_metric(gdist, ndist).item()
        
        graph.edges[(src, dst)]["distortion"] = distortion

    pos = nx.kamada_kawai_layout(graph)
        
    edges = graph.edges()
    
    distortions = [graph[u][v]['distortion'] for u, v in edges]
    
    fig, ax = plt.subplots()
    
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    ec = nx.draw_networkx_edges(graph, pos, width=3, edgelist=edges, edge_color=distortions, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax)
    nx.draw_networkx_nodes(graph, pos, node_size=30)
    
    fig.colorbar(ec, ax=ax, orientation='horizontal', pad=0.0, shrink=0.5)

    plt.axis('off')
    
    plt.show()
    
if __name__ == "__main__":       
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default=20, type=int, help="the dimension of manifold.")
    parser.add_argument("--graph", default='grid', type=str) 
    parser.add_argument("--model", default='euclidean', type=str)
    parser.add_argument("--metric", default='l1', type=str)
    parser.add_argument("--batch_size", default=2048, type=int)  
        
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    graph = DatasetLoader.get(args)
    
    id2node, triples, graph = DatasetLoader.build_triples(graph)       
                            
    args.num_points = len(id2node)
        
    if args.batch_size == -1:
        args.batch_size = len(triples)
    
    src_dst_ids = torch.LongTensor([(src, dst) for src, dst, _ in triples]).to(args.device)
    graph_distances = torch.Tensor([distance for _, _, distance in triples]).to(args.device)    
    
    model = DisModel(args).to(args.device)
    
    checkpoint_path = f'model.pt'
    
    model.load_state_dict(torch.load(checkpoint_path))
            
    model.eval()
    metric = AverageDistortionMetric()
    
    with torch.no_grad():
        norm_distances = model(src_dst_ids)
     
    distortion = metric.calculate_metric(graph_distances, norm_distances)
    
    avg_distortion = np.mean(distortion) * 100
    
    print(avg_distortion) # sanity-check

    plot_graph_edges(src_dst_ids, graph_distances, norm_distances, graph)


