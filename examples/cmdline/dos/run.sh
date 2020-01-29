#!/bin/bash
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

tbmodels dos -i ../eigenvals/input/silicon_model.hdf5 -k 10 10 10 -e -10 20 -n 1000 -s 0 -w 0.1 -o silicon_dos.hdf5
