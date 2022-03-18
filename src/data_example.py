from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils.convert as convert
import networkx as nx
G = nx.read_gml("../med_gmls/sin.gml")
data = convert.from_networkx(G)
data.foo = 10

print(data)


data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch.foo)
