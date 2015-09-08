#!/usr/bin/perl
# DARWIN UPDATE VERSION NUMBERS
# Stephen Gould <stephen.gould@anu.edu.au>
#

use strict;
use Getopt::Std;

my %opts = ();
getopts("hv:", \%opts);
if (defined($opts{h})) {
    print STDERR "USAGE: ./updateVersion.pl [-v <version>]\n\n";
    print STDERR "OPTIONS:\n";
    print STDERR "  -v \"<version>\" :: update to version string\n\n";
    print STDERR "EXAMPLE:\n";
    print STDERR "  ./updateVersion.pl -v \"<x>.<y>\"\n";
    print STDERR "  git commit -a -m \"updated version\"\n";
    print STDERR "  git push\n";
    print STDERR "  git tag <x>.<y>.0\n";
    print STDERR "  git push --tags\n";
    print STDERR "  (create release in GitHub)\n";
    print STDERR "  ./updateVersion.pl -v \"<x>.<y+1> (beta)\"\n";
    print STDERR "  git commit -a -m \"updated version\"\n";
    print STDERR "  git push\n";
    print STDERR "\n";
    exit(0);
}

# extract current version numbers
print STDERR "checking current version numbers...\n";

`grep "#define DRWN_VERSION" ../src/lib/base/drwnConstants.h` =~ m/\"(.*)\"/;
print "DRWN_VERSION: $1\n";

`grep "PROJECT_NUMBER" ../src/doc/doxygen.conf` =~ m/=\s*(\S.*)/;
print "PROJECT_NUMBER: $1\n";

`grep "DRWNLIBMAJORVER = " ../make.mk` =~ m/(\d+)/;
print "DRWNLIBMAJORVER: $1\n";
`grep "DRWNLIBMINORVER = " ../make.mk` =~ m/(\d+\.\d+)/;
print "DRWNLIBMINORVER: $1\n";

# change to new version number
if (defined($opts{v})) {
    print STDERR "updating to version $opts{v}...\n";

    `sed -i 's/#define DRWN_VERSION.*/#define DRWN_VERSION   "$opts{v}"/' ../src/lib/base/drwnConstants.h`;
    `sed -i 's/PROJECT_NUMBER.*=.*/PROJECT_NUMBER = $opts{v}/' ../src/doc/doxygen.conf`;

    if ($opts{v} =~ m/^(\d+)\.(\d+(\.\d+)?)/) {
        my $major = $1;
        my $minor = $2;
        $minor .= ".0" unless ($minor =~ m/\./);
        `sed -i 's/DRWNLIBMAJORVER.*=.*/DRWNLIBMAJORVER = $major/' ../make.mk`;
        `sed -i 's/DRWNLIBMINORVER.*=.*/DRWNLIBMINORVER = $minor/' ../make.mk`;
    }
}
