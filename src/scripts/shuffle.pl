#!/usr/bin/perl
# ----------------------------------------------------------------------------
# DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
# Distributed under the terms of the BSD license (see the LICENSE file)
# Copyright (c) 2007-2015, Stephen Gould
# All rights reserved.
#
# ----------------------------------------------------------------------------
# FILENAME:    shuffle.pl
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
# DESCRIPTION:
#   Shuffles the lines in a file or standard input.
#
# ----------------------------------------------------------------------------

@lines = <>;
fisher_yates_shuffle(\@lines);
print join("", @lines);

# from perl coodbook (4.17 randomizing an array)
sub fisher_yates_shuffle {
    my $array = shift;
    my $i;
    for ($i = @$array; --$i; ) {
        my $j = int rand ($i+1);
        next if $i == $j;
        @$array[$i,$j] = @$array[$j,$i];
    }
}
