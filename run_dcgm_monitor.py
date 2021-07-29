""" Run the dcgm_monitor

Example Usage:
sudo python3 run_dcgm_monitor.py
"""

from hfta.workflow.dcgm_monitor import DcgmMonitor, dcgm_monitor_start, dcgm_monitor_stop

monitor = DcgmMonitor('a100')
monitor_thread = dcgm_monitor_start(monitor, '/home/ubuntu/dcgm_monitor_out')
# Do stuff
dcgm_monitor_stop(monitor, monitor_thread)

