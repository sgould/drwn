#!/bin/csh
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2014, Stephen Gould
# All rights reserved.
#

if ($#argv == 0) then
    set files = "*.graph.xml.gz"
else
    set files = "$*"
endif

foreach i ( $files )
    set baseName = $i:r:r:r
    echo "uncompressing ${baseName}..."
    gunzip ${baseName}.graph.xml.gz
    echo "running inference on ${baseName}..."
    ../../bin/rosettaInference -profile -verbose -log ${baseName}.log ${baseName}.graph.xml
    echo "compressing ${baseName}..."
    gzip ${baseName}.graph.xml
end
