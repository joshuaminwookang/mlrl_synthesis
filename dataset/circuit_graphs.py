import networkx as nx
import json
import argparse
import glob, os.path, os
import subprocess


def run_yosys(verilog_path, output, ip, graph_type, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        if graph_type == "gl":
            p = subprocess.check_call(['yosys', '-p', "read_verilog {}; proc; memory; opt; techmap; opt; gml -o {}/{}.gml".format(str(verilog_path), str(output),ip)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        else :
            p = subprocess.check_call(['yosys', '-p', "read_verilog {}; opt; gml -o {}/{}.rtl.gml".format(str(verilog_path), str(output),ip)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        return True
    except:
        return False


def run_yosys_from_dir(load_dir_path,output_path,graph_type):
    verilogs = glob.glob(os.path.normpath(os.path.join(os.getcwd(), load_dir_path+"/*.v")))
    for verilog_file in verilogs:
        ip = verilog_file[verilog_file.rindex('/')+1:verilog_file.find('.v')]
        run_yosys(verilog_file, output_path, ip, graph_type)
        if graph_type == "gl":
            M = nx.read_gml("{}/{}.gml".format(output_path,ip))
            G = nx.Graph(M)
            nx.write_gml(G, "{}/{}.gml".format(output_path,ip))
        else:
            M = nx.read_gml("{}/{}.rtl.gml".format(output_path,ip))
            G = nx.Graph(M)
            nx.write_gml(G, "{}/{}.rtl.gml".format(output_path,ip))

def summary(result_dir):
    filename = os.path.basename(os.path.dirname(result_dir))
    gmls = glob.glob(os.path.normpath(os.path.join(result_dir,"*.gml")))
    gmls.sort()
    print(gmls)
    results = {}
    for gml_file in gmls:
        ip = gml_file[gml_file.rindex('/')+1:gml_file.find('.gml')]
        G = nx.read_gml(gml_file)
        results[ip] = (len(G.nodes), len(G.edges))
    json_file = filename+'_summary.json'
    with open(json_file, 'w') as outfile:
        json.dump(results, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, required=True, help="verilog file path")
    parser.add_argument('-o','--output', type=str, required=True, help="verilog file path")
    parser.add_argument('--graph_type', type=str, default="gl", help="gl or rtl")
    parser.add_argument('-s','--summary', action='store_true')
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    graph_type = args.graph_type
    if args.summary:
        summary(input_dir)
    else:
        run_yosys_from_dir(input_dir,output_dir,graph_type)
        summary(output_dir)
    


if __name__ == "__main__":
    main()
