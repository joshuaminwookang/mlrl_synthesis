#!/usr/bin/python3

import argparse
import os,random

aig_ind_ops = ["&dc2", "&syn2", "&syn3", "&syn4", "&b", "&b -d",
               "&if -W 300 -x -K 6", "&if -W 300 -g -K 6", "&if -W 300 -y -K 6"]
aig_ch_ops = ["&synch2", "&dch", "&dch -f"]
abc9_ops = aig_ind_ops + aig_ch_ops + ["&if -W 300 -K 6 -v;&mfs;&st", "&if -W 300 -K 6 -v;&st"]
abc9_ops_lib = aig_ind_ops + aig_ch_ops + ["&if -W 300 -v;&mfs;&st", "&if -W 300 -v;&st"]

# old stuff
options = ["rewrite", "rewrite -z", "refactor", "refactor -z", "resub -K 8", "resub -K 4", "resub -K 12", "resub -N 0", "resub -N 2", "resub -N 3", "balance",  "dc2"]

opener = "strash;ifraig;scorr;"
# closure_ftune = "strash;ifraig;scorr;dc2;strash;dch -f;if -K 6;mfs2;lutpack -S 1"
closure_whitebox_delay = "strash;ifraig;scorr;strash;dch -f;if -v;mfs2;print_stats -l"
closure = "dretime; strash; dch -f; if -v; mfs2" # LUTPACK or not; dretime or not with -

def synthesize(output_dir, index):
    run_name = "test_{}".format(index)
    run_output_file = os.path.join(output_dir, run_name+".txt")
    xdc_file = os.path.join(output_dir, run_name+".xdc")
    if os.path.exists(run_output_file):
        print("reusing cached test" + run_name)
        return
    tcl_script = ''' 
    set_param generalmaxThreads 1 
    set_property IS_ENABLED 0 [get_drc_checks {PDRC-43}]
    if {[file exists "$(dirname ${path})/${ip}_vivado.tcl"] == 1} {
        source ${ip}_vivado.tcl
    } else {
        read_verilog $(basename ${path%.gz})
        #read_verilog ${path}
    }
    if {[file exists "$(dirname ${path})/${ip}.top"] == 1} {
    set fp [open $(dirname ${path})/${ip}.top]
    set_property TOP [string trim [read \$fp]] [current_fileset]
    } else {
    set_property TOP [lindex [find_top] 0] [current_fileset]
    }
    cd ${pwd}
    read_xdc -unmanaged ${xdc_file}
    synth_design -part ${xl_device} -mode out_of_context ${SYNTH_DESIGN_OPTS}
    opt_design -directive Explore 
    '''
    with open(xdc_file, 'w') open as f:
        f.write(tcl_script)
def main():
    parser = argparse.ArgumentParser(
            description='Single run of Yosys-ABC + Vivado')
    parser.add_argument('--input', type=str, help='Input Verilog')
    parser.add_argument('--device', type=str, help='Target Xilinx FPGA device', default="xc7a200")
    parser.add_argument('--speed', type=int, help='Target clock rate (in picoseconds) for syntehsis', default=5000)
    parser.add_argument('--grade', type=int, help='Target Xilinx FPGA device grade', default=1)
    parser.add_argument('--abc9', type=int, help='Index of current sequence', default=-1)

    parser.add_argument('--idx', type=int, help='Index of current sequence', default=-1)
    parser.add_argument('--random', type=int, help='Random Sequence Length', default=0)
    parser.add_argument('--vivado', type=int, help='Random Sequence Length', default=0)
    parser.add_argument('--lut_library', type=int, help='Index of LUT Library to use: 0 is default', default=0)
    parser.add_argument('--stochastic', type=int, help='Whether to use stochastic synthesis', default=0)
    args = parser.parse_args()

    synth="yosys-abc9"
    doABC9 = args.abc9 > 0
    doRandom = args.random_seq_len > 0
    doStochastic = args.stochastic > 0
    lutLib = args.lut_library
    xilinxDevice = "xc7a200tffv1156-1"

    output_dir="tab_{1}_{2}_{3}_{4}".format(synth, ip, dev, idx)
    if doRandom:
        output_dir="tab_{1}_{2}_{3}_random_{4}".format(synth, ip, dev, idx)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    print("rec_start3 " + os.path.dirname(os.path.abspath(__file__)) + "/include/rec6Lib_final_filtered3_recanon.aig")
    if args.in_idx == -2:
        print("&scorr;&sweep;&dc2;&dch -f;&if -W 300 -K 6 -v;&mfs;")
        return
    if lut_lib_num > 0:
        print("read_lut " + os.path.dirname(os.path.abspath(__file__)) + "/lut_library/LUTLIB_{}.txt".format(lut_lib_num))
    if do_stochastic:
        print(get_abc9_stochastic(args.in_idx, lut_lib_num))
        return
    if do_random:
        #print(get_rand_seq_abc9(args.random_seq_len,lut_lib_num, args.in_idx))
        random_num = random.randint(len(abc9_ops)**(args.random_seq_len-1), len(abc9_ops)**(args.random_seq_len))
        print(get_seq_abc9_single_list(random_num, lut_lib_num))
        print("&ps; &pfeatures stats.json; &pfanstats fanstats.json;&write temp.aig")
    elif do_abc9:
        print(get_seq_abc9_single_list(args.in_idx, lut_lib_num))
        print("&ps; &pfeatures stats.json; &pfanstats fanstats.json;&write temp.aig")
    else :
        print(get_seq(args.in_idx))
    # print("write_blif internal.blif")

if __name__ == '__main__':
    main()    
    
