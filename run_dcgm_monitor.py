""" Run the dcgm_monitor

Example Usage:
sudo python3 run_dcgm_monitor.py
"""
import time
from hfta.workflow.dcgm_monitor import DcgmMonitor, dcgm_monitor_start, dcgm_monitor_stop

monitor = DcgmMonitor('a100')
print('Starting...')
monitor_thread = dcgm_monitor_start(monitor, '/home/ubuntu/dcgm_monitor_out')
time.sleep(2) # dcgm_monitor will sample every x seconds.
# Do stuff
metrics = dcgm_monitor_stop(monitor, monitor_thread)
print(metrics)
print('Done.')