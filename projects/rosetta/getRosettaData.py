#!/usr/bin/python3
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2017, Stephen Gould
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
    try:
        urllib.request.urlretrieve(u, fname)
    except:
        print("ERROR: could not download data file " + fname)
        exit()

# untar files
print("decompressing data files...")
tarfile.open(fname, "r:gz").extractall(".")

# fetch sparse class
fname = "sparse_cell_2.tgz"
if (os.path.isfile(fname) == 0):
    print("fetching sparse cell class...")
    u = "http://cyanover.fhcrc.org/sparse_cell_2.tgz"
    try:
        urllib.request.urlretrieve(u, fname)
        tarfile.open(fname, "r:gz").extractall(".")
    except:
        print("ERROR: could not download data file " + fname)
        exit()
