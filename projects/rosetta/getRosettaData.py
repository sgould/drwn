#!/usr/bin/python3
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2014, Stephen Gould
# All rights reserved.
#

import os.path
import urllib.request
import tarfile

# fetch data from web
fname = "Rosetta_Design_Dataset.tgz"
if (os.path.isfile(fname) == 0):
    print("WARNING: this make take a while...")
    u = "http://jmlr.csail.mit.edu/papers/volume7/yanover06a/Rosetta_Design_Dataset.tgz"
    urllib.request.urlretrieve(u, fname)

# untar files
tarfile.open(fname, "r:gz").extractall(".")

# fetch sparse class
fname = "sparse_cell_2.tgz"
if (os.path.isfile(fname) == 0):
    print "fetching sparse cell class..."
    u = "http://cyanover.fhcrc.org/sparse_cell_2.tgz"
    urllib.request.urlretrieve(u, fname)
    tarfile.open(fname, "r:gz").extractall(".")
