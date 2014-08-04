#!/bin/csh
#
# DARWIN INSTALLATION OF MAC OSX ESSENTIAL SOFTWARE
# Copyright (c) 2007-2014, Stephen Gould
#
# FILENAME:    macosx.sh
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
#

set OS = `uname -s`
set CODEBASE = `pwd`/..
if ("$OS" != "Darwin") then
    exit 1
endif

# homebrew
if ( `/usr/bin/which -s brew || echo "1"` ) then
#    curl -fsSL https://raw.github.com/gist/323731 | /usr/bin/ruby || exit 1
#    curl -fksSL https://raw.github.com/mxcl/homebrew/master/Library/Contributions/install_homebrew.rb | /usr/bin/ruby || exit 1
#    curl -fsSL https://raw.github.com/mxcl/homebrew/go | /usr/bin/ruby || exit 1
    ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)" || exit 1
endif

# wget
if ( `/usr/bin/which -s wget || echo "1"` ) then
    /usr/local/bin/brew install wget
    alias wget '/usr/local/bin/wget'
endif

# pkg-config
if ( `/usr/bin/which -s pkg-config || echo "1"` ) then
    curl http://pkgconfig.freedesktop.org/releases/pkg-config-0.25.tar.gz -o pkg-config-0.25.tar.gz || exit 1
    tar -xf pkg-config-0.25.tar.gz || exit 1
    cd pkg-config-0.25
    ./configure CC="gcc -arch i386 -arch x86_64" CXX="g++ -arch i386 -arch x86_64" CPP="gcc -E" CXXCPP="g++ -E"
    make
    sudo make install
    cd -
endif

# cmake
if ( `/usr/bin/which -s cmake || echo "1"` ) then
    curl -O http://www.cmake.org/files/v2.8/cmake-2.8.6.tar.gz
    tar zxvf cmake-2.8.6.tar.gz
    cd cmake-2.8.6
    ./bootstrap
    make
    sudo make install
    cd -
endif
