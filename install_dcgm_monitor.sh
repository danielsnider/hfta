# Install DCGM
set -v
set -e
architecture=x86_64
distribution=[YOUR OS HERE eg. "ubuntu2004"]
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$architecture /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$architecture/7fa2af80.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$architecture/cuda-$distribution.pin
sudo mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm

# # Install HFTA
# git clone https://github.com/danielsnider/hfta
# cd hfta
# pip install -e .

# # Stop this process because dcgm_monitor.py wants to start it
# pkill nv-hostengine

# # Run example dcgm_monitor
# sudo python3 run_dcgm_monitor.py

