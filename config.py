
import argparse

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default=4, type=int, help="the dimension of manifold.")
    parser.add_argument("--graph", default='tree', type=str) 
    parser.add_argument("--model", default='euclidean', type=str)
    parser.add_argument("--metric", default='f1', type=str)

    parser.add_argument("--batch_size", default=-1, type=int) 
    
    parser.add_argument("--gnn", default='gcn', type=str) 
    parser.add_argument("--patience", default=200, type=int) 
    parser.add_argument("--learning_rate", default=0.01, type=float) 
    parser.add_argument("--weight_decay", default=0, type=float)

    parser.add_argument("--epoch", default=500, type=int)

    parser.add_argument("--r", default=2., type=float, help='fermi-dirac decoder parameter for lp')

    parser.add_argument("--t", default=1., type=float, help='fermi-dirac decoder parameter for lp')

    parser.add_argument("--val_every", default=1, type=int)
    
    parser.add_argument("--tree_height", default=5, type=int)    

    parser.add_argument("--max_gradient_norm", default=250, type=float)    

    parser.add_argument("--grid_dim", default=5, type=int)    
    
    return parser.parse_args()