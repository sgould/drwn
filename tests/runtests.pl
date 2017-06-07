#!/usr/bin/perl -I ../external/perl5
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Copyright (c) 2007-2017, Stephen Gould
# All rights reserved.
# ----------------------------------------------------------------------------
# FILENAME:    runtests.pl
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
# DESCRIPTION:
#   Used to run regression tests. Use command line:
#     % ./runtest.pl [<OPTIONS>] <testFilename>
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
#   This script results that XML::Simple and Text::Diff be installed. To do so
#   run "sudo cpan", then "install XML::Simple" and "install Text::Diff".
# ----------------------------------------------------------------------------

use XML::Simple;
use Text::Diff;
use Getopt::Std;
use strict;

my $OSNAME = $^O;
my $OUTDIR = ($OSNAME eq "MSWin32") ? "$ENV{TEMP}\\" : "/tmp/";

my %opts = ();
getopts("kln:vx", \%opts);
if ($#ARGV != 0) {
    print STDERR "USAGE: ./runtest.pl [<OPTIONS>] <testFilename>\n";
    print STDERR "OPTIONS:\n";
    print STDERR "  -k        :: keep output files (in $OUTDIR) if test fails\n";
    print STDERR "  -l        :: list test names\n";
    print STDERR "  -n <name> :: only run test <name>\n";
    print STDERR "  -v        :: verbose output\n";
    print STDERR "  -x        :: don't run actual tests\n";
    exit(-1);
}

my $testFilename = $ARGV[0];
my $testList = XMLin($testFilename, KeyAttr => 1, 
		     ForceArray => ['test', 'command', 'file']);

# list test names
if (exists($opts{'l'})) {
    for (my $i = 0; $i <= $#{$testList->{test}}; $i++) {
        my $test = $testList->{test}->[$i];
        print "$test->{name}\n";
    }
    exit(0);
}

my @failedTests = ();

# run each test
for (my $i = 0; $i <= $#{$testList->{test}}; $i++) {
    print "----------------------------------------\n";
    my $test = $testList->{test}->[$i];
    # check if test is enabled and -n flag not set
    if ((($test->{enabled} ne "true") && !exists($opts{'n'})) ||
         (exists($opts{'n'}) && ($test->{name} ne $opts{'n'}))) {
	print "Skipping test \"$test->{name}\"\n";
	next;
    }
    
    # start the test
    my $passed = 1;
    print "Running test \"$test->{name}\"...\n";
    
    # add output path to any temporary files
    my $params = $test->{parameters};
    for (my $j = 0; $j <= $#{$test->{file}}; $j++) {
	my $binding = "\\\$" . ($j + 1);
	$params =~ s/$binding/${OUTDIR}$test->{file}->[$j]/g;
    }
    
    # construct the command line
    my $cmdline = "$test->{command} $params 1> ${OUTDIR}$test->{name}.stdout 2> ${OUTDIR}$test->{name}.stderr";
    $cmdline =~ s/\//\\/g if ($OSNAME eq "MSWin32");
    print "$cmdline\n";

    # check that output files don't already exist
    for (my $j = 0; $j <= $#{$test->{file}}; $j++) {
	if (-e "${OUTDIR}$test->{file}->[$j]") {
	    print STDERR "ERROR: output file $test->{file}->[$j] already exists\n";
	    push @failedTests, $test->{name};
	    $passed = 0;
	    last;
	}
    }
    next if (!$passed);
    
    # run the command
    if (!exists($opts{'x'})) {
	`$cmdline`;
	if (exists($opts{'v'})) {
	    print `cat ${OUTDIR}$test->{name}.stdout`;
	    print `cat ${OUTDIR}$test->{name}.stderr`;
	}
	if (!exists($test->{ignoreStdout}) || ($test->{ignoreStdout} ne "true")) {
	    #print `diff output/$test->{name}.stdout ${OUTDIR}$test->{name}.stdout`;
	    #$passed &= !$?;
	    my $d = "MISSING GOLDSTANDARD RESULTS\n";
            if (-e "output/$test->{name}.stdout") {
                $d = diff("output/$test->{name}.stdout", "${OUTDIR}$test->{name}.stdout");
            }
	    print $d;
	    $passed &= ($d eq "");
	}
	if (!exists($test->{ignoreStderr}) || ($test->{ignoreStderr} ne "true")) {
	    my $d = "MISSING GOLDSTANDARD RESULTS\n";
            if (-e "output/$test->{name}.stderr") {
                $d = diff("output/$test->{name}.stderr", "${OUTDIR}$test->{name}.stderr");
            }
	    print $d;
	    $passed &= ($d eq "");
	}
	for (my $j = 0; $j <= $#{$test->{file}}; $j++) {
	    my $d = diff("output/$test->{file}->[$j]", "${OUTDIR}$test->{file}->[$j]");
	    print $d;
	    $passed &= ($d eq "");
	}
    }

    print "...test \"$test->{name}\" ";
    if ($passed) {
	print "PASSED\n";
    } else {
	push @failedTests, $test->{name};
	print "FAILED\n";
    }

    # remove log and output files
    if ($passed || !exists($opts{'k'})) {
	unlink("${OUTDIR}$test->{name}.stdout");
	unlink("${OUTDIR}$test->{name}.stderr");
	for (my $j = 0; $j <= $#{$test->{file}}; $j++) {
	    unlink("${OUTDIR}$test->{file}->[$j]");
	}
    }
}

# print list of failed test
print "----------------------------------------\n";
if ($#failedTests >= 0) {
    print "FAILED TESTS:\n  ";
    print join("\n  ", @failedTests);
} else {
    print "ALL TESTS PASSED";
}
print "\n----------------------------------------\n";

