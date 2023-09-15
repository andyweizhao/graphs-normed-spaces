import time
import torch
from pytorchtools import EarlyStopping
from geoopt.optim import RiemannianAdam, RiemannianSGD
from config import load_arguments
import random
import os.path as osp
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric import seed_everything
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from sympa.model import LPModelGNN, LPModelGNNProductEuclidean
from torch_geometric.utils import negative_sampling, convert
from datasets import DatasetLoader

class Runner(object):
    def __init__(self, model, optimizer, train_data, val_data, test_data, args, src_dst_ids, graph_dists):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.src_dst_ids=src_dst_ids
        self.graph_dists=graph_dists
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args
        
    def run(self):
        
        checkpoint_path = f'model.pt'
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, path=checkpoint_path)    
        for epoch in range(1, self.args.epoch + 1):

            start = time.perf_counter()
            train_loss = self.train_epoch(self.train_data, self.src_dst_ids, self.graph_dists)
            exec_time = time.perf_counter() - start

            print(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                
            if epoch % self.args.val_every == 0:
                
                roc, ap = self.evaluate(self.val_data)

                early_stopping(-roc, self.model)
                if early_stopping.early_stop:
                    break                

        self.model.load_state_dict(torch.load(checkpoint_path))
        
        roc, ap = self.evaluate(self.test_data)
        
        print(f"Final Results: ROC: {roc * 100:.2f}, Precision: {ap * 100:.2f}") 
        
    def train_epoch(self, train_data, src_dst_ids, graph_dists):
 
        self.model.train()
        self.optimizer.zero_grad()

        x = self.model.encode(train_data.x, train_data.edge_index)
        
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
    
    
        preds = self.model.decode(x, edge_label_index).view(-1)
        loss_lp = self.criterion(preds, edge_label)
        
        loss = loss_lp
            
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, eval_data):

        self.model.eval()
        x = self.model.encode(eval_data.x, eval_data.edge_index)
        preds = self.model.decode(x, eval_data.edge_label_index).view(-1)

        if preds.is_cuda:
            labels = eval_data.edge_label.cpu().numpy()
            preds = preds.detach().cpu().numpy()
        
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        return roc, ap       
    
def setup_envs(seed=-1):

    if seed == -1: seed = random.randint(1, 1000)
    seed_everything(seed)
        
    args = load_arguments()    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
  
    return args 

def build_dataloader(args):

    transform = T.Compose([
        T.ToDevice(args.device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True,
                          add_negative_train_samples=False,
                          neg_sampling_ratio=1.0,
                        )
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    graph = Planetoid(path, name=args.graph, transform=transform)

    return graph 

def build_batchloader(triples, args):
    src_dst_ids = torch.LongTensor([(src, dst) for src, dst, _ in triples]).to(args.device)
    graph_dists = torch.Tensor([distance for _, _, distance in triples]).to(args.device)    

    return src_dst_ids, graph_dists


if __name__ == "__main__":    
    args = setup_envs(seed=42)
    
    graph = build_dataloader(args)
    
    train_data, val_data, test_data = graph[0]
    args.num_features = graph.num_features
    args.num_points = train_data.num_nodes
    
    id2node, triples, _ = DatasetLoader.build_triples(convert.to_networkx(train_data))
    src_dst_ids, graph_dists = build_batchloader(triples, args)
    
    print(args)
    
    if args.model == 'prod-eueu':
        model = LPModelGNNProductEuclidean(args).to(args.device)
    else:
        model = LPModelGNN(args).to(args.device)
    
    optimizer = RiemannianAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=None)

    runner = Runner(model, optimizer, args=args,
                    train_data=train_data, val_data=val_data, test_data=test_data, src_dst_ids=src_dst_ids, graph_dists=graph_dists)
    runner.run()