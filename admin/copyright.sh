#!/bin/bash
# DARWIN COPYRIGHT NOTICE UPDATE SCRIPT
# Stephen Gould <stephen.gould@anu.edu.au>
#

RUN=0
VERBOSE=0
while getopts "xv" opt; do
    case "$opt" in
        x) 
            RUN=1
            ;;
        v)
            VERBOSE=1
            ;;
        esac
done

YEAR=`date +"%Y"`

for i in include projects src tests
do
    for j in cpp h py pl dox xml m
    do
        # find files with copyright notice
        if [ $VERBOSE -eq 1 ]
        then
            find ../${i} -name "*.${j}" -exec grep -H "Copyright (c) 2007-20\?\?" {} \;
        else
            COUNT=`find ../${i} -name "*.${j}" -exec grep -H "Copyright (c) 2007-20\?\?" {} \; | wc -l`
            echo "...changing ${COUNT} files macthing ${i}/*.${j}"
        fi
        
        # change copyright notice
        if [ $RUN -eq 1 ]
        then
            find ../${i} -name "*.${j}" -exec sed -i "s/2007-20../2007-${YEAR}/" {} \;
        fi
    done
done
