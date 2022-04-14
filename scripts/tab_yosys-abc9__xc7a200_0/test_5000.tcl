# generated at Tue Apr 12 21:26:28 PDT 2022
set_param general.maxThreads 1
set_property IS_ENABLED 0 [get_drc_checks {PDRC-43}]
read_edif .edif
read_xdc -unmanaged test_5000.xdc
link_design -part xc7a200tffv1156-1 -mode out_of_context -top 
#report_timing_summary
report_design_analysis
report_utilization
# For now, don't do place and route
# place_design -directive Explore
# route_design -directive Explore
# report_utilization
# report_timing -no_report_unconstrained
# report_clocks
# report_design_analysis
# report_power
# report_io
