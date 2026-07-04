#!/bin/bash
cd /home/santhi/Documents/Jarvis
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jarvis
echo ""
echo "Starting JARVIS Web Server..."
echo "Open this in your browser: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
python web_jarvis.py
