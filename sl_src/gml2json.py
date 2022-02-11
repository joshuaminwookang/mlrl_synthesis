import networkx as nx
import json
import argparse
import glob, os.path, os


def gmls_from_dir(load_dir_path, output_dir):
    gmls = glob.glob(os.path.normpath(os.path.join(os.getcwd(), load_dir_path+"/*.gml")))
    for gml_file in gmls:
        G = nx.read_gml(gml_file)
        Gp = nx.relabel.convert_node_labels_to_integers(G)

        edges =list(Gp.edges())
        nodes = list(Gp.nodes.data())
        nodes_dict = { n[0]:n[1] for n in nodes}
        data = {}
        data['edges'] = edges
        data['features'] = nodes_dict
        tag = gml_file[gml_file.rindex('/')+1:gml_file.find('.gml')]
        json_file = os.path.normpath(os.path.join(output_dir, tag + ".json"))
        with open(json_file, 'w') as outfile:
            json.dump(data, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="gml file path")
    parser.add_argument('--output', type=str, required=True, help="json file path")


    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gmls_from_dir(input_dir, output_dir)
    


if __name__ == "__main__":
    main()
