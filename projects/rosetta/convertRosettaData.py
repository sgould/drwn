#!/usr/bin/python3
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2015, Stephen Gould
# All rights reserved.
#

import glob
import os
import gzip

for i in glob.glob("*.dee.mat"):
    baseName = i[:-8]
    print("converting " + baseName + "...")
    os.system("matlab -nojvm -nosplash -r \"rosetta2drwn('" + baseName + "'); exit;\"")

    print("compressing " + baseName)
    with open(baseName + '.graph.xml', 'rb') as f_in:
        with gzip.open(baseName + '.graph.xml.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    os.remove(baseName + ".graph.xml")
