#!/bin/csh
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2014, Stephen Gould
# All rights reserved.
#

# fecth data from web
if (! -e Rosetta_Design_Dataset.tgz) then
    echo "WARNING: this make take a while..."
    wget http://jmlr.csail.mit.edu/papers/volume7/yanover06a/Rosetta_Design_Dataset.tgz
endif

# untar files
tar zxvf Rosetta_Design_Dataset.tgz
