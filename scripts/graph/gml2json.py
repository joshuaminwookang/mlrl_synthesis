import networkx as nx
import json
import argparse
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="gml file path")
parser.add_argument('--output', type=str, required=True, help="json file path")


args = parser.parse_args()
gml_file = args.input
json_file = args.output

G = nx.read_gml(gml_file)
Gp = nx.relabel.convert_node_labels_to_integers(G)

edges =list(Gp.edges())
nodes = list(Gp.nodes.data())
nodes_dict = {}
for n in nodes:
    nodes_dict[n[0]] = n[1]
# print(nodes_dict)
data = {}
data['edges'] = edges
data['features'] = nodes_dict
with open(json_file, 'w') as outfile:
    json.dump(data, outfile)
