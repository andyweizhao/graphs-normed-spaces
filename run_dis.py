import time
import torch
import numpy as np
from sympa.eval.metrics import AverageDistortionMetric, MeanAveragePrecisionMetric
from torch.utils.data import TensorDataset
from pytorchtools import EarlyStopping
from datasets import DatasetLoader, BatchGenerator
from geoopt.optim import RiemannianAdam
from sympa.model import DisModel
from config import load_arguments
import random
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything

import warnings
warnings.filterwarnings("ignore")

def distortion_loss(graph_distances, manifold_distances, batch_size = 2048):
    
    num_points = manifold_distances.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size
    
    total_loss = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_points)
    
        batch_manifold_distances = manifold_distances[start_idx:end_idx]
        batch_graph_distances = graph_distances[start_idx:end_idx]
    
        batch_loss = torch.pow(torch.div(batch_manifold_distances, batch_graph_distances), 2)
        batch_loss = torch.abs(batch_loss - 1)
        batch_loss = batch_loss.sum()
    
        total_loss += batch_loss
    
    return total_loss

class Runner(object):
    def __init__(self, model, optimizer, scheduler, id2node, train_loader, valid_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.id2node = id2node
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metric = AverageDistortionMetric()
        self.args = args
        
    def run(self):
        checkpoint_path = f'model.pt'
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, path=checkpoint_path)    
        for epoch in range(1, self.args.epoch + 1):

            start = time.perf_counter()
            train_loss = self.train_epoch(self.train_loader, epoch)
            exec_time = time.perf_counter() - start

            print(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                
            if epoch % self.args.val_every == 0:
                distortion = self.evaluate(self.valid_loader)
                print(f"Results ep {epoch}: tr loss: {train_loss:.1f}, "
                              f"val avg distortion: {distortion * 100:.2f}")                
                
                early_stopping(distortion, self.model)
                if early_stopping.early_stop:
                    break                

        self.model.load_state_dict(torch.load(checkpoint_path))
        
        distortion = self.evaluate(self.valid_loader)
        precision = self.calculate_mAP()
        
        print(f"Final Results: Distortion: {distortion * 100:.2f}, Precision: {precision * 100:.2f}")        
                                     
    def train_epoch(self, train_split, epoch_num):
        tr_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_split):
            src_dst_ids, graph_distances = batch
            
            manifold_distances = self.model(src_dst_ids)
            
            loss = distortion_loss(graph_distances, manifold_distances)
            loss.backward()

            tr_loss += loss.item()
            
            clip_grad_norm_(self.model.parameters(), args.max_gradient_norm)
            
            self.optimizer.step()
        
        torch.cuda.empty_cache()
        return tr_loss / train_split.size
                             
    def evaluate(self, eval_split):
        self.model.eval()
        total_distortion = []
        for batch in eval_split:
            src_dst_ids, graph_distances = batch
            with torch.no_grad():
                manifold_distances = self.model(src_dst_ids)
                distortion = self.metric.calculate_metric(graph_distances, manifold_distances)
            total_distortion.extend(distortion.tolist())

        avg_distortion = np.mean(total_distortion)
        return avg_distortion    
    
    def calculate_mAP(self):
        distance_matrix = self.build_distance_matrix()
        
        dataset = TensorDataset(self.valid_loader.src_dst_ids, self.valid_loader.graph_dist)
        mAP = MeanAveragePrecisionMetric(dataset)
        
        return mAP.calculate_metric(distance_matrix)

    def build_distance_matrix(self):
        all_nodes = torch.arange(0, len(self.id2node)).unsqueeze(1)
        distance_matrix = torch.zeros((len(all_nodes), len(all_nodes)))
        self.model.eval()
        for node_id in range(len(self.id2node)):
            src = torch.LongTensor([[node_id]]).repeat(len(all_nodes), 1)
            src[node_id] = (node_id + 1) % len(all_nodes)   # hack to not ask for the distance to the same value
            batch = torch.cat((src, all_nodes), dim=-1)
            with torch.no_grad():
                distances = self.model(batch)
            distances[node_id] = 0
            distance_matrix[node_id] = distances.view(-1)
        return distance_matrix        

def setup_envs(seed=-1):

    if seed == -1: seed = random.randint(1, 42)
    seed_everything(seed)
        
    args = load_arguments()    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    return args 

def build_batchloader(triples, args):
    train_src_dst_ids = torch.LongTensor([(src, dst) for src, dst, _ in triples]).to(args.device)
    train_distances = torch.Tensor([distance for _, _, distance in triples]).to(args.device)    
    
    valid_src_dst_ids = train_src_dst_ids
    valid_distances = train_distances
    
    train_loader = BatchGenerator(train_src_dst_ids, train_distances, args.batch_size)
    valid_loader = BatchGenerator(valid_src_dst_ids, valid_distances, args.batch_size)    
    
    return train_loader, valid_loader

if __name__ == "__main__":    
    args = setup_envs(seed=42)
        
    graph = DatasetLoader.get(args)
    id2node, triples, graph = DatasetLoader.build_triples(graph)
    args.num_points = len(id2node)
    
    if args.batch_size == -1:
        args.batch_size = len(triples)
    
    print(args)
    
    train_loader, valid_loader = build_batchloader(triples, args)
    
    model = DisModel(args).to(args.device)

    optimizer = RiemannianAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=None)
    
    runner = Runner(model, optimizer, None, id2node=id2node, args=args,
                    train_loader=train_loader, valid_loader=valid_loader)
    runner.run()
