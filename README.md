# NeurIPS23 

Implementation for `Normed Spaces for Graph Embeddings`

## Available Spaces

- `Euclidean-l1`, `Euclidean-l2`, `Euclidean-linf`
- `Poincare`, `Lorentz`, `Sphere`
- `Poincare x Euclidean-l1`, `Poincare x Euclidean-l2`, `Poincare x Euclidean-linf`
- `Upper-riem`, `Upper-f1`, `Upper-finf`, `Bounded-riem`, `Bounded-f1`, `Bounded-finf`
- `SPD`

## Usage 

Below are the instructions about running experiments in the l1 and linf normed spaces.

``python run_dis.py --model euclidean --metric l1 --dims 20 --learning_rate 0.001 --batch_size 2048 --epoch 3000 --graph grid``

``python run_dis.py --model euclidean --metric linf --dims 20 --learning_rate 0.01 --batch_size 2048 --epoch 3000 --graph tree``

``python run_dis.py --model prod-eueu --metric l1,linf --dims 20 --learning_rate 0.01 --batch_size 2048 --epoch 3000 --graph tree``

``python run_lp_gnn.py --dims 64 --graph cora --model euclidean --metric l1 --gnn gcn --learning_rate 0.01 --batch_size -1 --epoch 1000``

``python run_lp_gnn.py --dims 64 --graph cora --model prod-eueu --metric l1,linf --gnn gcn --learning_rate 0.01 --batch_size -1 --epoch 1000``

Below is the way to visualize embedding distortion in the linf normed space.

``python visual.py --model euclidean --metric linf --dims 20 --graph grid`` 
    
## Datasets 

- `Grid` `Tree`  `TreexTree` `Tree o Grid`  `Grid o Tree`  `Fullerenes`
- `Margulis`  `Paley`  `Chordal`
- `USCA312`  `Biseasome`  `CSPHD` `EuroRoad`  `Facebook`
- `ML-100k` `LastFM`  `MeetUp-NYC`
- `Cora` `Citeseer`

## Requirements
- Python == 3.8
- scikit-learn == 1.0.1 
- torch == 1.12.1
- torch-geometric == 2.1.0
- geoopt == 0.5.0
- networkit == 10.0
- networkx == 2.6.3

## TODO
- [ ] Recommender Systems
