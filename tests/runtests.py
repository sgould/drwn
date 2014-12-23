#!/usr/bin/python
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2014, Stephen Gould
# All rights reserved.
# ----------------------------------------------------------------------------
# FILENAME:    runtests.py
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
# DESCRIPTION:
#   Used to run regression tests. Use command line:
#     % python runtest.py [<OPTIONS>] <testFilename>
#
#   Assumes <testfile> is an XML file with the following entities:
#
#   <test name="TEST NAME"
#     enabled="TRUE OR FALSE"
#     command="FULL PATH TO EXECUTABLE"
#     parameters="PARAMETERS WITH $<n> FOR OUTPUT FILENAME SUBSTITUTIONS"
#     ignoreStdout="TRUE OR FALSE"
#     ignoreStderr="TRUE OF FALSE"
#   >
#   <file>FILE TO SUBSTITUTE FOR FILE 1</file>
#   <file>FILE TO SUBSTITUTE FOR FILE 2</file>
#   ...
#  </test>
#
# ----------------------------------------------------------------------------

import sys, getopt, os, re
import xml.etree.ElementTree as etree

# usage ----------------------------------------------------------------------

def usage():
    sys.stderr.write("USAGE: ./runtest.py [<OPTIONS>] <testFilename>\n")
    sys.stderr.write("OPTIONS:\n")
    sys.stderr.write("  -k        :: keep output files (in $OUTDIR) if test fails\n")
    sys.stderr.write("  -l        :: list test names\n")
    sys.stderr.write("  -n <name> :: only run test <name>\n")
    sys.stderr.write("  -v        :: verbose output\n")
    sys.stderr.write("  -x        :: don't run actual tests\n")

# main -----------------------------------------------------------------------

try:
    opts, args = getopt.getopt(sys.argv[1:], "kln:vx")
except getopt.GetoptError:
    usage()
    exit(-1)

if len(args) != 1:
    usage()
    exit(-1)

testFilename = args[0]
testList = etree.parse(testFilename).getroot()

# list test names
if ('-l', '') in opts:
    for t in testList:
        print t.attrib['name']
    exit()

runOnlyTests = []
for o in opts:
    if o[0] == '-n':
        runOnlyTests.append(o[1])

failedTests = []

# run each test
for t in testList:
    print "----------------------------------------"
    # check if test is enabled and -n flag not set
    if (((t.attrib['enabled'] != "true") and (len(runOnlyTests) == 0)) or
        ((len(runOnlyTests) > 0) and (t.attrib['name'] not in runOnlyTests))):
        print "Skipping test " + t.attrib['name']
        continue

    # start the test
    testPassed = True;
    print "Running test " + t.attrib['name'] + "..."

    # add output path to any temporary files
    try:
        params = t.attrib['parameters']
    except:
        params = ""

    if ('file' in t.attrib):
        for i in range(0, len(file)):
            binding = "\\$" + str(i + 1)
            re.sub(binding, file[i], params)

    # construct the command line
    cmdline = t.attrib['command'] + " " + params + " 1> " + t.attrib['name'] + ".stdout 2> " + t.attrib['name'] + ".stderr"
    print cmdline

    # check that output files don't already exist
    if 'file' in t.attrib:
        for f in t.attrib['file']:
            if os.path.exist(f):
                sys.stderr.write("ERROR: output file " + f + " already exists\n")
                failedTests.append(t.attrib['name'])
                testPassed = False;
                break

    if not testPassed:
        continue
    
    # run the command
    if ('-x', '') in opts:
        # TODO
        testPassed = False
        pass

    if testPassed:
        print "...test \"" + t.attrib['name'] + "\" PASSED"
    else:
	failedTests.append(t.attrib['name'])
        print "...test \"" + t.attrib['name'] + "\" FAILED"

    # remove log and output files
    if testPassed or (('-k', '') not in opts):
        pass
        # TODO
	#unlink("${OUTDIR}$test->{name}.stdout");
	#unlink("${OUTDIR}$test->{name}.stderr");
	#for (my $j = 0; $j <= $#{$test->{file}}; $j++) {
	#    unlink("${OUTDIR}$test->{file}->[$j]");
	#}


# print list of failed test
print "----------------------------------------"
if len(failedTests) > 0:
    print "FAILED TESTS:"
    for t in failedTests:
        print "  " + t
else:
    print "ALL TESTS PASSED"
print "----------------------------------------"

