#!/usr/bin/python3
# 2021-22 Josh Kang (mkang@eecs.berkeley)
# 
import argparse
from multiprocessing import Pool
import os,random, subprocess, glob, json, re
import abc_scripts

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

# run Yosys with specified synthesis method (ASIC/FPGA mapping + custom ABC/ABC9 sequence)
def run_yosys(input_file, run_name, index, synth_method, random_seq_len, cwd=os.getcwd(), stdout=None, stderr=None):
    yosys_file, ops_list = gen_yosys_script(cwd, input_file, run_name, synth_method, index, random_seq_len)
    try:
        log_file_path = os.path.join(cwd, 'yosys.log')
        print("Running Yosys: {}".format(run_name))
        p = subprocess.check_call(['yosys', yosys_file, '-l', log_file_path], \
                                  cwd=cwd, stdout=subprocess.DEVNULL, stderr=stderr)

        #print("yosys {} -l {}".format(yosys_file, log_file_path))
        return True, ops_list
    except:
        return False, []
    
# helper: write Yosys script based on synth method
def gen_yosys_script(output_sub_dir, verilog_path, run_name, synth_method, index,random_seq_len):
    yosys_file = os.path.join(output_sub_dir, "{}.ys".format(run_name))
    abc_script_file = os.path.join(output_sub_dir, "{}.abc".format(run_name))
    synth_script = ""
    abc_script_string = ""
    # TODO: add for ASIC mapping 
    if synth_method == "fpga-abc9":
        synth_script =  "synth_xilinx -dff -flatten -noiopad -abc9 -edif {0}.edif -script {0}.abc".format(run_name)
        abc_script_string = abc_scripts.get_abc9_sequence(index, random_seq_len)
    else:
        synth_script =  "synth_xilinx -dff -flatten -noiopad -edif {0}.edif -script {0}.abc".format(run_name)
        abc_script_string = abc_scripts.get_abc_sequence(index, random_seq_len)
    script = '''read_verilog {0}\n{1}\n'''.format(verilog_path, synth_script)
    with open(yosys_file, 'w') as f:
        f.write(script)
        
    with open(abc_script_file, 'w') as f:
        f.write(abc_script_string)

    return yosys_file, abc_script_file

# run Vivado for FPGA design analysis
# generate necessary scripts (TCL and XDC file) 
def run_vivado_da(run_name, clock_period, cwd=os.getcwd(), stdout=None, stderr=None):
    tcl_file = gen_vivado_tcl(cwd, run_name)
    xdc_file = gen_vivado_xdc(cwd, run_name, clock_period)
    try:
        log_file_path = os.path.join(cwd, 'analysis.log')
        print("Running Vivado: {}".format(run_name))
        #print("vivado -nojournal -log {} -mode batch -source {}".format(log_file_path, tcl_file))
        p = subprocess.check_call(['vivado', '-nojournal','-log', log_file_path, '-mode', 'batch', '-source', tcl_file], \
                                  cwd=cwd, stdout=subprocess.DEVNULL, stderr=stderr)
        return True, log_file_path
    except:
        return False, log_file_path
    
# helper: write Vivado TCL script for post-synth analysis
# TODO: additional TCL generation for Vivado synthesis -> design analysis
def gen_vivado_tcl(output_sub_dir, run_name):
    tcl_file = os.path.join(output_sub_dir, run_name+".tcl")
    script = '''set_param general.maxThreads 1
set_property IS_ENABLED 0 [get_drc_checks {{PDRC-43}}]
read_edif {0}.edif
read_xdc -unmanaged {0}.xdc
link_design -part xc7a200tffv1156-1 -mode out_of_context -top {0}
report_design_analysis
report_utilization
'''.format(run_name)
    with open(tcl_file, 'w') as f:
        f.write(script)
    return tcl_file

# helper function: write Vivado XDC file
def gen_vivado_xdc(output_sub_dir, run_name, clock_period):
    xdc_file = os.path.join(output_sub_dir, run_name+".xdc")
    clock_ns = "{:.2f}".format(clock_period/1000)
    script = '''# Auto-generated XDC file; read with read_xdc -unmanaged
if {[llength [get_ports -quiet -nocase -regexp .*cl(oc)?k.*]] != 0} {
  create_clock -period 5.00 [get_ports -quiet -nocase -regexp .*cl(oc)?k.*]
} else {
  puts "WARNING: Clock constraint omitted because expr \"[get_ports -quiet -nocase -regexp .*cl(oc)?k.*]\" matched nothing."
}
'''
    with open(xdc_file, 'w') as f:
        f.write(script)
    return xdc_file

def gen_vivado_yosys_summary(output_sub_dir, ip, index, abc_script, vivado_log):
    data = {}
    data['Index'] = index
    data['Benchmark'] = ip
    try:
        with open(abc_script, "r") as fp:
            scripts = fp.readlines()
            data['Sequence'] = scripts[2][:-2]
            print(data['Sequence'])
    except OSError:
        print("Could not open/read file:", abc_script)
    try:
        with open(vivado_log, "r") as fp:
            data["Path_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Path Delay", fp)[0])[0])
            fp.seek(0)
            data["Slice_LUTs"] = int(re.findall(r'\d+', lines_that_contain("Slice LUTs", fp)[0])[0])
    except OSError:
        print("Could not open/read file:", vivado_log)

    summary_file = os.path.join(output_sub_dir, "{}_{}.json".format(ip, index))
    with open(summary_file, "w") as f:
        json.dump(data, f, indent=4)
    
    
# run method to run synthesis + post-synthesis analysis on a single design
# TODO: add Vivado/Synopsis synthesis for FPGA and ASIC mapping
def run_synthesis(input_file=None, output_dir=None, index=0, synth_method='fpga-abc9', clock_period=5000, grade=1, device="",\
                  run_analysis=True, do_random=False, random_seq_len=0):
    # convert file paths to absolute paths
    output_dir = os.path.abspath(output_dir)
    input_file = os.path.abspath(input_file)
    
    # parse input file name 
    filename = os.path.basename(input_file)
    ip = filename[filename.find('_') + 1 :filename.find('.v')] #TODO: add for VHDL or other file formats (if needed)

    # create output directory as needed
    output_sub_dir=os.path.join(output_dir, "tab_{0}_{1}_{2}".format(synth_method, ip, index))
    if do_random:
        output_sub_dir=os.path.join(output_dir, "tab_{0}_{1}_random_{2}".format(synth_method, ip, index))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_sub_dir):
        os.mkdir(output_sub_dir)

    # bookkeeping: output files 
    run_name = "{}_{}".format(ip, index)
    run_output_file = os.path.join(output_sub_dir, run_name+".json")
    edif_file = os.path.join(output_sub_dir, run_name+".edif") #TODO: add in-place parsing of post-synthesis QoR?

    # if results exist, skip
    if os.path.exists(run_output_file):
        print("reusing cached results from: " + run_name)
        return 0
    
    # if EDIF file exists, synthesis was already run on current IP and synth sequence
    # skip script generation and Yosys-ABC run
    if os.path.exists(edif_file):
        print("reusing cached {}".format(edif_file))
        abc_script_file = os.path.join(output_sub_dir, "{}.abc".format(run_name))
    else:
        success, abc_script_file = run_yosys(input_file, run_name, index, synth_method, random_seq_len, cwd=output_sub_dir)
        if not success :
            print("ERROR in Yosys run for {} {}".format(run_name, synth_method))
            return 1

    # run commercial tool for post-Synthesis design analysis
    if run_analysis:
        if synth_method == "fpga-abc9" or synth_method == "fpga-abc":
            success, analysis_log = run_vivado_da(run_name, clock_period, cwd=output_sub_dir)
            if not success :
                print("ERROR in Vivado design run for {} {}".format(run_name,synth_method))
                return 1
            gen_vivado_yosys_summary(output_sub_dir, ip, index, abc_script_file, analysis_log)
    return 0



def run_batch_synthesis(**kwargs):
    # assume that input path is a directory of all input Verilogs
    params = kwargs
    batch_size = params['batch']
    #min_len = params['min_length']
    max_len = params['max_length']
    del params['batch']
    #del params['min_length']
    del params['max_length']
    verilogs = glob.glob(os.path.normpath(kwargs['input_file']+"/*.v"))
    print(verilogs)
    
    if params['synth_method'] == "fpga-abc9":
        end_idx  = abc_scripts.get_index_bounds_abc9(max_len)
    else:
        end_idx  = abc_scripts.get_index_bounds_abc(max_len)

    # spawn parallel jobs(processes)
    for v in verilogs:
        index = 0
        while True:
            if end_idx <= index :
                break
            pool = Pool(processes=batch_size)
            for i in range(batch_size):
                if end_idx <= index :
                    continue
                params['input_file'] = v
                params['index'] = index
                done = pool.apply_async(run_synthesis, list(params.values()))
                index = index +1
            pool.close()
            pool.join()
        
def test(**kwargs):
    max_len = kwargs['max_length']
    if kwargs['synth_method'] == "fpga-abc9":
        end_idx  = abc_scripts.get_index_bounds_abc9(max_len)
    else:
        end_idx  = abc_scripts.get_index_bounds_abc(max_len)
    # for i in range(end_idx):
    #     print(abc_scripts.parse_index_abc(i-1))
    print(end_idx)
    
def main():
    parser = argparse.ArgumentParser(
            description='Single run of Yosys-ABC + Vivado')
    # Single run parameters
    parser.add_argument('-i','--input_file' , type=str, help='Input Verilog') # TODO Batch mode
    parser.add_argument('-o','--output_dir' , type=str, help='Output directory top level') # TODO Batch mode
    parser.add_argument('--index', type=int, help='Index of current sequence', default=-1)
    parser.add_argument('--synth_method', type=str, help='Target Mapping + synth method', default="fpga-abc9")
    parser.add_argument('--clock_period', type=int, help='Target clock rate (in picoseconds) for syntehsis', default=5000)
    parser.add_argument('--grade', type=int, help='Target Xilinx FPGA device grade', default=1)
    parser.add_argument('--device', type=str, help='Target Xilinx FPGA device', default="xc7a200tffv1156-1")
    parser.add_argument('--run_analysis', type=bool, help='Run Vivado or Synopsis post-Synthesis analysis backend', default=True)
    parser.add_argument('--do_random', action='store_true', help='Test random synthesis recipes')

    # Batch run parameters
    parser.add_argument('-b', '--batch', type=int, help='Synthesis run batch size', default=0)
    parser.add_argument('-m', '--max_length', type=int, help='maximum length of synthesis recipes to run', default=0)
    #parser.add_argument('-n', '--min_length', type=int, help='minimum length of synthesis recipes to run', default=0)

    #parser.add_argument('--stochastic', type=int, help='Whether to use stochastic synthesis', default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    if args.batch > 0:
        test(**kwargs)
        #run_batch_synthesis(**kwargs)
    else:
        kwargs = vars(args)
        kwargs.pop('batch')
        kwargs.pop('max_length')
        #kwargs.pop('min_length')
        run_synthesis(**kwargs)
    
if __name__ == '__main__':
    main()    
    
