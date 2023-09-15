
import argparse

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default=4, type=int, help="the dimension of manifold.")
    parser.add_argument("--graph", default='mupnyc', type=str) 
    parser.add_argument("--model", default='bounded', type=str)
    parser.add_argument("--metric", default='f1', type=str)
    parser.add_argument("--loss", default='bce', type=str)
     
    parser.add_argument("--batch_size", default=2048, type=int) 
    parser.add_argument("--learning_rate", default=0.01, type=float) 
    
    parser.add_argument("--weight_decay", default=0, type=float)

    parser.add_argument("--epoch", default=500, type=int)


    parser.add_argument("--neg_sample_size", default=1, type=int, help="Neg sample to use in loss.")
    parser.add_argument("--hinge_margin", default=1, type=float, help="Margin for hinge loss.")
    
    parser.add_argument("--reduce_factor", default=5, type=float, help="Factor to reduce lr on plateau.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--patience", default=50, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--burnin", default=10, type=int, help="Number of initial epochs to train with reduce lr.")
        
    return parser.parse_args()

  
    