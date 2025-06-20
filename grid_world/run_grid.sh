#!/bin/bash

AGGREGATION="dsum"
if [ ! -z "$1" ]; then
  AGGREGATION=$1
fi

python3 grid.py --aggregation $AGGREGATION
