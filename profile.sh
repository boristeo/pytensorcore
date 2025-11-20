#/usr/bin/env sh

/opt/nvidia/nsight-compute/2025.2.1/ncu --call-stack-type native --call-stack-type python --import-source=yes --set=full -f --export=profile.ncu-rep ./run.py
