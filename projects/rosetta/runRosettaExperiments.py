#!/usr/bin/python3
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2016, Stephen Gould
# All rights reserved.
#

import os
import gzip
import glob
import sys

if (len(sys.argv) == 1):
    files = glob.glob("*.graph.xml.gz")
else:
    files = sys.argv[1:]

for i in files:
    baseName = i.split('.')[0]

    print("uncompressing " + baseName + "...")
    with gzip.open(baseName + '.graph.xml.gz', 'rb') as f_in:
        with open(baseName + '.graph.xml', 'wb') as f_out:
            f_out.writelines(f_in)

    print("running inference on " + baseName + "...")
    os.system("../../bin/rosettaInference -profile -verbose -log " +
        baseName + ".log " + baseName + ".graph.xml")

    print("removing uncompressed " + baseName + "...")
    os.remove(baseName + ".graph.xml")
