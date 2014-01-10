#!/bin/csh
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2014, Stephen Gould
# All rights reserved.
#

foreach i ( *.dee.mat )
    set baseName = $i:r:r
    echo "converting ${baseName}..."
    matlab -nojvm -nosplash -r "rosetta2drwn('$baseName'); exit;"
    echo "compressing ${baseName}..."
    gzip ${baseName}.graph.xml
end
