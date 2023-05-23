import networkx as nx
from networkx.generators import expanders
import networkit as nk
import numpy as np

class DatasetLoader(object):    
    
    @staticmethod    
    def load_expanders(args):
        if args.graph == "expander-margulis":
            graph = expanders.margulis_gabber_galil_graph(25)
        elif args.graph == "expander-chordal":
            graph = expanders.chordal_cycle_graph(523)
        elif args.graph == "expander-paley":
            graph = expanders.paley_graph(101)
        return graph    
    
    @staticmethod    
    def load_real_graph(args):
        path = f"data/{args.graph}.edgelist"
        graph = nx.Graph(name=args.graph)
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split()
                if len(line) == 2 or (len(line) > 2 and not line[2].replace(".", "", 1).isdigit()):
                    graph.add_edge(line[0], line[1])
                else:
                    graph.add_edge(line[0], line[1], weight=float(line[2]))
        return graph
                        
    @classmethod
    def get(self, args):        
        graph_name = args.graph    
        if graph_name == "grid":
            graph = nx.grid_graph(dim=(5,5,5,5))
            
        elif graph_name == "tree":
            tree_valency = 3
            tree_height = 5 
            graph = nx.balanced_tree(tree_valency, tree_height)
                                
        elif graph_name == "treextree":
            tree = nx.balanced_tree(2,3)
            graph = nx.cartesian_product(tree, tree)
            
        elif graph_name == "treexgrid":
            tree = nx.balanced_tree(2,3)
            grid = nx.grid_graph(dim=(4,4))
            graph = nx.cartesian_product(tree, grid)
            
        elif graph_name == "gridotree":
            tree = nx.balanced_tree(2,4)
            grid = nx.grid_graph(dim=(5,5))
            graph = nx.algorithms.operators.rooted_product(grid, tree, list(tree.nodes())[0])

        elif graph_name == "treeogrid":
            tree = nx.balanced_tree(2,4)
            grid = nx.grid_graph(dim=(5,5))
            graph = nx.algorithms.operators.rooted_product(tree, grid, list(grid.nodes())[0])

        elif graph_name == "complete":
            complete_n = args.complete_n
            graph = nx.complete_graph(n=complete_n)        
          
        elif 'expander' in args.graph : 
            graph = self.load_expanders(args)

        else: graph = self.load_real_graph(args)            
        
        print("Generated graph:", graph_name, "#V:", graph.number_of_nodes(), "#E:", graph.number_of_edges())
              
        return graph
    
    @staticmethod
    def build_triples(graph):
        """
        Builds triples of (src, dst, distance) for each node in the graph, to all other connected nodes.
        PRE: distances in the graph are symmetric
        :param graph: networkx graph
        :return: set of triples
        """
        id2node = {i: node for i, node in enumerate(sorted(graph.nodes()))}        
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
                
        print(nx.info(graph))
        
        if nx.is_weighted(graph):
            gk = nk.nxadapter.nx2nk(graph, weightAttr="weight")
            distance_type = float
        else:
            gk = nk.nxadapter.nx2nk(graph)
            distance_type = int
        shortest_paths = nk.distance.APSP(gk).run().getDistances()
        n_nodes = len(shortest_paths)
        UNREACHABLE_DISTANCE = 1e10     # nk sets a very large distance value (~1e308) for unreachable nodes
    
        triples, pairs = set(), set()
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                distance = shortest_paths[i][j]
                if 0 < distance < UNREACHABLE_DISTANCE:
                    if (j, i) not in pairs:  # checks that the symmetric triplets is not there
                        pairs.add((i, j))
                        triples.add((i, j, distance_type(distance)))
        return id2node, triples, graph

class BatchGenerator(object):
    """Generate batches of data of a specified size."""
    
    def __init__(self, src_dst_ids, graph_dist, batch_size):
        self.src_dst_ids = src_dst_ids
        self.graph_dist = graph_dist
        self.batch_size = batch_size
        self.size = len(src_dst_ids)
        
    def __iter__(self):
        self.n_batches = int(np.ceil(self.size / self.batch_size))  
        self.batch_idx = 0
        return self
    
    def __next__(self):
        if self.batch_idx >= self.n_batches:
            raise StopIteration
        
        batch_start = self.batch_idx * self.batch_size
        batch_end = (self.batch_idx + 1) * self.batch_size
        
        batch_src_dst_ids = self.src_dst_ids[batch_start:batch_end]
        batch_graph_dist = self.graph_dist[batch_start:batch_end]
        
        self.batch_idx += 1
        
        return batch_src_dst_ids, batch_graph_dist