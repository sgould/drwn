#!/bin/csh
#
# DARWIN INSTALLATION OF EXTERNAL LIBRARIES
# Copyright (c) 2007-2015, Stephen Gould
#
# FILENAME:    install.sh (wxWidgets | Eigen)
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
#

set OS = `uname -s`
set CODEBASE = `pwd`/..

# wxWidgets
if (! -e wx && (("$1" == "wxWidgets") || ("$1" == "wx"))) then
    set WXBUILD = "wxWidgets"
    set VERSION = "3.0.2"
    if (! -e ${WXBUILD}-${VERSION}) then
        if (! -e ${WXBUILD}-${VERSION}.tar) then
	    if (! -e ${WXBUILD}-${VERSION}.tar.bz2) then
		wget http://prdownloads.sourceforge.net/wxwindows/${WXBUILD}-${VERSION}.tar.bz2 || exit 1
	    endif
	    bunzip2 ${WXBUILD}-${VERSION}.tar.bz2 || exit 1
	endif
        tar xvf ${WXBUILD}-${VERSION}.tar || exit 1
        rm -f ${WXBUILD}-${VERSION}.tar.bz2
    endif
    cd ${WXBUILD}-${VERSION}
    mkdir -p build
    cd build || exit 1
    if ("$OS" == "Darwin") then
	../configure --with-opengl --enable-monolithic --with-osx_cocoa --with-macosx-version-min=10.7 \
	    --disable-webkit --disable-webviewwebkit --enable-maxosx_arch=x86_64 CC=clang CXX=clang++
	make || exit 1
    else
	../configure --disable-shared --with-opengl --enable-monolithic
        make || exit 1
    endif
    cd ../..
    ln -s ${WXBUILD}-${VERSION} ${CODEBASE}/external/wx
endif

# eigen
if (! -e Eigen && (("$1" == "Eigen") || ("$1" == "eigen"))) then
    set VERSION = "3.2.7"
    wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/${VERSION}.tar.bz2 -O eigen-${VERSION}.tar.bz2 || exit 1
    bunzip2 eigen-${VERSION}.tar.bz2 || exit 1
    tar xvf eigen-${VERSION}.tar
    if (-d eigen-eigen-${VERSION}) then
        mv eigen-eigen-${VERSION} eigen-${VERSION}
    else if (-d eigen-eigen-bdd17ee3b1b3) then # 3.2.5
        mv eigen-eigen-bdd17ee3b1b3 eigen-${VERSION}
    else if (-d eigen-eigen-b30b87236a1b) then # 3.2.7
        mv eigen-eigen-b30b87236a1b eigen-${VERSION}
    else
        echo "*** COULD NOT DETERMINE EIGEN DIRECTORY NAME ***"
        exit 1
    endif
    rm -f eigen-${VERSION}.tar
    ln -s eigen-${VERSION}/Eigen ${CODEBASE}/external/Eigen
endif

# opencv
if (! -e opencv && (("$1" == "OpenCV") || ("$1" == "opencv"))) then
    set VERSION = "2.4.10"
    if (! -e opencv-${VERSION}) then
        if (! -e opencv-${VERSION}.tar.gz) then
            wget -c https://github.com/Itseez/opencv/archive/${VERSION}.tar.gz
        endif
        tar zxvf ${VERSION}.tar.gz
    endif
    cd opencv-${VERSION}
    if (`uname` == Darwin) then
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	      -D CMAKE_INSTALL_PREFIX=${CODEBASE}/external/opencv \
	      -D CMAKE_CXX_FLAGS="-stdlib=libc++" \
	      -D CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++" \
	      -D BUILD_NEW_PYTHON_SUPPORT=OFF .
    else
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	      -D CMAKE_INSTALL_PREFIX=${CODEBASE}/external/opencv \
	      -D BUILD_NEW_PYTHON_SUPPORT=OFF .
    endif
    make
    make install
    cd ..
endif

if ("$1" == "OpenCV-trunk") then
    if (! -e OpenCV-trunk) then
        svn co https://code.opencv.org/svn/opencv/trunk OpenCV-trunk
    endif
    cd OpenCV-trunk
    svn update
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=${CODEBASE}/external/opencv \
          -D BUILD_NEW_PYTHON_SUPPORT=OFF .
    make
    make install
    cd ..
endif

# zlib
if (! -e zlib && "$1" == "zlib") then
    set VERSION = "1.2.8"
    wget -c http://zlib.net/zlib-${VERSION}.tar.gz || exit 1
    tar zxvf zlib-${VERSION}.tar.gz
    ln -s zlib-${VERSION} ${CODEBASE}/external/zlib
    cd zlib
    setenv CC gcc
    ./configure -t --shared
    make
    cd ..
endif

# lua
if (! -e lua && "$1" == "lua") then
    set VERSION = "5.2.4"
    wget -c http://www.lua.org/ftp/lua-${VERSION}.tar.gz || exit 1
    tar xzvf lua-${VERSION}.tar.gz
    cd lua-${VERSION}
    if ("$OS" == "Darwin") then
        make macosx
    else
        make linux
    endif
    make install INSTALL_TOP=${CODEBASE}/external/lua
endif
