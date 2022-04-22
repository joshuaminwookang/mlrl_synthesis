import networkx as nx
import json
import argparse
import glob, os.path, os
import subprocess


def run_yosys(verilog_path, output, ip, graph_type, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        if graph_type == "gl":
            p = subprocess.check_call(['yosys', '-p', "read_verilog {}; opt; techmap; opt; gml -o {}/{}.gl.gml".format(str(verilog_path), str(output),ip)], \
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
            M = nx.read_gml("{}/{}.gl.gml".format(output_path,ip))
            G = nx.Graph(M)
            nx.write_gml(G, "{}/{}.gl.gml".format(output_path,ip))
        else:
            M = nx.read_gml("{}/{}.rtl.gml".format(output_path,ip))
            G = nx.Graph(M)
            nx.write_gml(G, "{}/{}.rtl.gml".format(output_path,ip))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="verilog file path")
    parser.add_argument('--output', type=str, required=True, help="verilog file path")
    parser.add_argument('--graph_type', type=str, default="gl", help="gl or rtl")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    graph_type = args.graph_type
    run_yosys_from_dir(input_dir,output_dir,graph_type)
    


if __name__ == "__main__":
    main()
