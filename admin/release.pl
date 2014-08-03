#!/usr/bin/perl
# DARWIN RELEASE SCRIPT
# Stephen Gould <stephen.gould@anu.edu.au>
#

use strict;
use Getopt::Std;
use File::Path qw(remove_tree);
use File::Copy::Recursive qw(fcopy dircopy);
use File::Find ();

my %opts = ();
getopts("hv:xz", \%opts);
if (defined($opts{h})) {
    print STDERR "USAGE: ./release.pl [-v <version>] [-x] [-z]\n\n";
    print STDERR "OPTIONS:\n";
    print STDERR "  -v <version>  :: override release version\n";
    print STDERR "  -x            :: remove existing release directory\n";
    print STDERR "  -z            :: create compressed archives\n";
    print STDERR "DESCRIPTION\n";
    print STDERR "  Release script for Darwin source code. Make sure to check latest\n";
    print STDERR "  revision on Windows, Mac OS X, and Linux before official release.\n";
    print STDERR "  Also ensure that the changelog and previous release documentation\n";
    print STDERR "  is updated.\n\n";
    exit(0);
}

`grep "#define DRWN_VERSION" ../src/lib/base/drwnConstants.h` =~ m/\"(.*)\"/;

my $releaseVersion = $1;
if (defined($opts{v})) {
    print STDERR "WARNING: overriding version $releaseVersion with $opts{v}\n";
    $releaseVersion = $opts{v};
}

my $releaseName = "darwin.${releaseVersion}";
$releaseName =~ s/\s+/-/g;
$releaseName =~ s/[^\w.-]//g;

print "Releasing Darwin version $releaseVersion as $releaseName\n";
if (-d $releaseName) {
    if (defined($opts{x})) {
        print STDERR "WARNING: removing existing directory $releaseName\n";
        remove_tree($releaseName, {verbose => 0});
    } else {
        print STDERR "ERROR: directory $releaseName already exists (use -x to delete)\n";
        exit(1);
    }
}

# create release directory and copy source
print "Copying source code...\n";
mkdir($releaseName);
mkdir("${releaseName}/bin");
mkdir("${releaseName}/external");
mkdir("${releaseName}/include");
mkdir("${releaseName}/projects");
mkdir("${releaseName}/src");

fcopy("../INSTALL", ${releaseName});
fcopy("../LICENSE", ${releaseName});
fcopy("../Makefile", ${releaseName});
fcopy("../make.mk", ${releaseName});

dircopy("../external/win32", "${releaseName}/external/win32") or die $!;
dircopy("../external/rapidxml", "${releaseName}/external/rapidxml") or die $!;
fcopy("../external/install.sh", "${releaseName}/external") or die $!;
fcopy("../external/macosx.sh", "${releaseName}/external") or die $!;
dircopy("../include", "${releaseName}/include") or die $!;
dircopy("../src", "${releaseName}/src") or die $!;

# copy projects
my @PROJECTS = ('gui', 'nnGraph', 'matlab', 'multiSeg', 'patchMatch', 'photoMontage', 'rosetta', 'tutorial');
for (my $i = 0; $i <= $#PROJECTS; $i++) {
    dircopy("../projects/$PROJECTS[$i]", "${releaseName}/projects/$PROJECTS[$i]") or die $!;
}

# remove unnessessary subdirectories
print "Removing unnessessary subdirectories...\n";
remove_tree("${releaseName}/src/lib/_intermediates");

# remove subversion directories
print "Removing Subversion directories...\n";
File::Find::find({wanted => \&wanted, no_chdir => 1}, ${releaseName});
sub wanted {
    if (/\/\.svn$/s) {
        #print "removing $_\n";
        remove_tree($_, {verbose => 0});
    }
};

# clean
print "Cleaning build files...\n";
chdir(${releaseName});
system("make clean");
chdir("..");

# create archive
if (defined($opts{z})) {
    print "Creating zip and gz archives...\n";
    system("zip -r ${releaseName}.zip ${releaseName}");
    system("tar zcvf ${releaseName}.tar.gz ${releaseName}");
    remove_tree(${releaseName});
} else {
    print "WARNING: not creating archives (use -z to create)...\n";
}

