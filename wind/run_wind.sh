#!/bin/bash

RECURSIVE="max"
if [ ! -z "$1" ]; then
  RECURSIVE=$1
fi

python3 wind_all.py --RECURSIVE_TYPE $RECURSIVE
