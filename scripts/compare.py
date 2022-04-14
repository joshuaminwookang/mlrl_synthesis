import networkx as nx
import json
import argparse
import glob, os.path, os
import subprocess

def summary():
    print(os.path.normpath(os.path.join(os.getcwd(),"*.gml")))
    gmls = glob.glob(os.path.normpath(os.path.join(os.getcwd(),"*.gml")))
    print(gmls)
    results = {}
    for gml_file in gmls:
        ip = gml_file[gml_file.rindex('/')+1:gml_file.find('.gml')]
        G = nx.read_gml(gml_file)
        results[ip] = (len(G.nodes), len(G.edges))
        print(len(G.nodes), len(G.edges))
    json_file = 'summary.json'
    with open(json_file, 'w') as outfile:
        json.dump(results, outfile)


# G = nx.read_gml('adder1.gml')
# H = nx.read_gml('adder2.gml')

# print(len(G.nodes), len(G.edges))
# print(len(H.nodes), len(H.edges))
def main():
    summary()
    


if __name__ == "__main__":
    main()
