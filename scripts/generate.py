import networkx as nx
import json
import argparse
import glob, os.path, os
import subprocess

def run_yosys(verilog_path, ip, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['yosys', '-p', "read_verilog {}; opt; techmap; opt; gml -o {}_gl.gml".format(str(verilog_path), ip)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        return True
    except:
        return False


def run_yosys_from_dir(load_dir_path):
    verilogs = glob.glob(os.path.normpath(os.path.join(os.getcwd(), load_dir_path+"/*.v")))
    for verilog_file in verilogs:
        ip = verilog_file[verilog_file.rindex('/')+1:verilog_file.find('.v')]
        run_yosys(verilog_file, ip)
        # json_file = os.path.normpath(os.path.join(output_dir, tag + ".json"))
        # with open(json_file, 'w') as outfile:
        #     json.dump(data, outfile)


# G = nx.read_gml('adder1.gml')
# H = nx.read_gml('adder2.gml')

# print(len(G.nodes), len(G.edges))
# print(len(H.nodes), len(H.edges))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="verilog file path")
    args = parser.parse_args()
    input_dir = args.input
    run_yosys_from_dir(input_dir)
    


if __name__ == "__main__":
    main()
