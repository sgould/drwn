#!/usr/bin/perl
# BMP2XPM Bitmap to XPM conversion

for (my $i = 0; $i <= $#ARGV; $i++) {
    if ($ARGV[$i] =~ m/\.bmp/) {
        my $base = $`;
        print "$base\n";
        `convert ${base}.bmp -depth 8 -colors 16 ${base}.xpm`;
        `sed -i 's/\\[\\]/_xpm\\[\\]/' ${base}.xpm`;
    }
}
