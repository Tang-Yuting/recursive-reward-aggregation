#!/bin/bash

RECURSIVE="max"
if [ ! -z "$1" ]; then
  RECURSIVE=$1
fi

python3 run_wind.py --RECURSIVE_TYPE $RECURSIVE
