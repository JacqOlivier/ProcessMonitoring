#!/bin/bash
for file in *.json
do
  python3 /home/juc/dev/ProcessMonitoring/processmonitoring/run.py "$file"
done

