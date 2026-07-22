#!/bin/bash

module load python
module load cdo

CONFIG=config_remap.yml

python remap.py --config $CONFIG