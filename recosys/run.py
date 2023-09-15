import os
import random
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from rsmodel import RecoSys
from rslosses import HingeLoss, BCELoss
from rsmetrics import RankingBuilder, rank_to_metric
import time
from torch_geometric import seed_everything
from pytorchtools import EarlyStopping
import numpy as np
from geoopt.optim import RiemannianAdam, RiemannianSGD
from config import load_arguments


class Runner(object):
    def __init__(self, model, optimizer, scheduler, train, dev, test, samples, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.dev = dev
        self.test = test
        self.samples = samples
        self.loss = self.get_loss(args.loss)(ini_neg_index=0, end_neg_index=args.num_points, args=args)
        self.ranking_builder = RankingBuilder(ini_index=0, end_index=args.n_items, samples=samples)
        self.args = args
    
    def get_loss(self, loss):
        if loss == "bce": return BCELoss
        if loss == "hinge": return HingeLoss
        raise ValueError(f"Unrecognized loss: {loss}")
    
    def run(self):
        
        checkpoint_path = f'model.pt'
       
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, path=checkpoint_path)  

        for epoch in range(1, self.args.epoch + 1):
            self.set_burnin_lr(epoch)
            start = time.perf_counter()
            train_loss = self.train_epoch(self.train, epoch)
            exec_time = time.perf_counter() - start

            print(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')

            if epoch % self.args.val_every == 0:
                hitrate, ndcg = self.evaluate(self.dev)
                test_hitrate, test_ndcg = self.evaluate(self.test)
                
                print(f"Results ep {epoch}: HR@10: {hitrate:.2f}, nDCG: {ndcg:.3f}",
                      f"test_HR@10: {test_hitrate:.2f}, test_nDCG: {test_ndcg:.3f}")
                
                self.scheduler.step(hitrate)
                
                early_stopping(-hitrate, self.model)
                if early_stopping.early_stop:
                    break 

        self.model.load_state_dict(torch.load(checkpoint_path))

        hitrate, ndcg = self.evaluate(self.test)

        print(f"Final Results: HR@10: {hitrate:.2f}, nDCG: {ndcg:.3f}")    

    def train_epoch(self, train_split, epoch_num):
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_split):
            loss = self.loss.calculate_loss(self.model, batch[0])
            loss.backward()

            tr_loss += loss.item()
            gradient = self.model.embeddings.embeds.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            avg_grad_norm += grad_norm
            
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
        return tr_loss / len(train_split)

    def evaluate(self, eval_split):
        self.model.eval()
        ranking = []
        for batch in eval_split:
            with torch.no_grad():
                partial_ranking = self.ranking_builder.rank(self.model, batch[0])
                ranking.append(partial_ranking)

        ranking = np.concatenate(ranking, axis=0)
        hitrate, ndcg, mrr = rank_to_metric(ranking, at_k=10)

        return hitrate, ndcg

    def set_burnin_lr(self, epoch, burnin_factor=10):
        """Modifies lr if epoch is less than burn-in epochs"""
        if self.args.burnin < 1:
            return
        if epoch == 1:
            self.set_lr(self.get_lr() / burnin_factor)
        if epoch == self.args.burnin:
            self.set_lr(self.get_lr() * burnin_factor)

    def set_lr(self, value):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = value

    def get_lr(self):
        """:return current learning rate as a float"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
def setup_envs(seed=-1):

    if seed == -1: seed = random.randint(1, 42)
    seed_everything(seed)
        
    args = load_arguments()    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    return args 

def get_scheduler(optimizer, args):
    patience = round(args.patience / args.val_every)
    factor = 1 / float(args.reduce_factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, mode="max")

def load_data(args):
    data_path = f"prep/{args.graph}/{args.graph}.pickle"
    data = pickle.load(open(str(data_path), "rb"))

    train = TensorDataset(torch.LongTensor(data['train']).to(args.device))
    dev_data = data['dev'] if len(data['dev']) < 10000 else data['dev'][:10000]
    dev = TensorDataset(torch.LongTensor(dev_data).to(args.device))
    test = TensorDataset(torch.LongTensor(data['test']).to(args.device))
   
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)

    dev_loader = DataLoader(dev, sampler=SequentialSampler(dev), batch_size=args.batch_size)
    test_loader = DataLoader(test, sampler=SequentialSampler(test), batch_size=args.batch_size)
    
    return train_loader, dev_loader, test_loader, data["samples"], data


def get_quantities(data):
    n_users = len(data["id2uid"])
    n_items = len(data["id2iid"])
    return n_users, n_items, n_users + n_items

if __name__ == "__main__":    
    args = setup_envs(seed=42)
    train, dev, test, samples, data = load_data(args)
    n_users, n_items, n_entities = get_quantities(data)
    args.num_points = n_entities
    args.n_items = n_items
    
    print(args)
    rsmodel = RecoSys(args).to(args.device)
    
    optimizer = RiemannianSGD(rsmodel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=10)
    scheduler = get_scheduler(optimizer, args)
    
    runner = Runner(rsmodel, optimizer, scheduler, train=train, dev=dev, test=test, samples=samples, args=args)
    runner.run()







