# Auto-generated XDC file; read with read_xdc -unmanaged
if {[llength [get_ports -quiet -nocase -regexp .*cl(oc)?k.*]] != 0} {
  create_clock -period 5.00 [get_ports -quiet -nocase -regexp .*cl(oc)?k.*]
} else {
  puts "WARNING: Clock constraint omitted because expr \"[get_ports -quiet -nocase -regexp .*cl(oc)?k.*]\" matched nothing."
}
