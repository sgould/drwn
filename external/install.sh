#!/bin/bash
#
# DARWIN INSTALLATION OF EXTERNAL LIBRARIES
# Copyright (c) 2007-2018, Stephen Gould
#
# FILENAME:    install.sh (wxWidgets | Eigen | OpenCV)
# AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
#

OS=`uname -s`
CODEBASE=`pwd`/..

# wxWidgets
if [ ! -e wx ] && [ "$1" == "wxWidgets" -o "$1" == "wx" ]; then
    WXBUILD="wxWidgets"
    VERSION="3.0.4"
    if [ ! -e "${WXBUILD}-${VERSION}" ]; then
        if [ ! -e "${WXBUILD}-${VERSION}.tar" ]; then
	    if [ ! -e "${WXBUILD}-${VERSION}.tar.bz2" ]; then
                wget https://github.com/wxWidgets/wxWidgets/releases/download/v${VERSION}/${WXBUILD}-${VERSION}.tar.bz2 || exit 1
	    fi
	    bunzip2 ${WXBUILD}-${VERSION}.tar.bz2 || exit 1
	fi
        tar xvf ${WXBUILD}-${VERSION}.tar || exit 1
        rm -f ${WXBUILD}-${VERSION}.tar.bz2
    fi
    cd ${WXBUILD}-${VERSION}
    mkdir -p build
    cd build || exit 1
    if [ "$OS" == "Darwin" ]; then
	../configure --with-opengl --enable-monolithic --with-osx_cocoa --with-macosx-version-min=10.7 \
	    --disable-webkit --disable-webviewwebkit --enable-maxosx_arch=x86_64 CC=clang CXX=clang++
	make || exit 1
    else
	../configure --disable-shared --with-opengl --enable-monolithic
        make || exit 1
    fi
    cd ../..
    ln -s "${WXBUILD}-${VERSION}" "${CODEBASE}/external/wx"
fi

# eigen
if [ ! -e Eigen ] && [ "$1" == "Eigen" -o "$1" == "eigen" ]; then
    VERSION="3.2.10"
    wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/${VERSION}.tar.bz2 -O eigen-${VERSION}.tar.bz2 || exit 1
    bunzip2 eigen-${VERSION}.tar.bz2 || exit 1
    tar xvf eigen-${VERSION}.tar
    if [ -d "eigen-eigen-${VERSION}" ]; then
        mv eigen-eigen-${VERSION} eigen-${VERSION}
    elif [ -d eigen-eigen-bdd17ee3b1b3 ]; then # 3.2.5
        mv eigen-eigen-bdd17ee3b1b3 eigen-${VERSION}
    elif [ -d eigen-eigen-b30b87236a1b ]; then # 3.2.7
        mv eigen-eigen-b30b87236a1b eigen-${VERSION}
    elif [ -d eigen-eigen-07105f7124f9 ]; then # 3.2.8
	mv eigen-eigen-07105f7124f9 eigen-${VERSION}
    elif [ -d eigen-eigen-b9cd8366d4e8 ]; then # 3.2.10
	mv eigen-eigen-b9cd8366d4e8 eigen-${VERSION}
    else
        echo "*** COULD NOT DETERMINE EIGEN DIRECTORY NAME ***"
        exit 1
    fi
    rm -f eigen-${VERSION}.tar
    ln -s eigen-${VERSION}/Eigen ${CODEBASE}/external/Eigen
fi

# opencv
if [ ! -e opencv ] && [ "$1" == "OpenCV" -o "$1" == "opencv" ]; then
    VERSION="2.4.10"
    if [ ! -e "opencv-${VERSION}" ]; then
        if [ ! -e "opencv-${VERSION}.tar.gz" ]; then
            wget -c https://github.com/Itseez/opencv/archive/${VERSION}.tar.gz
        fi
        tar zxvf ${VERSION}.tar.gz
    fi
    cd opencv-${VERSION}
    if [ `uname` == Darwin ]; then
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	      -D CMAKE_INSTALL_PREFIX=${CODEBASE}/external/opencv \
	      -D CMAKE_CXX_FLAGS="-stdlib=libc++" \
	      -D CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++" \
	      -D BUILD_NEW_PYTHON_SUPPORT=OFF .
    else
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	      -D CMAKE_INSTALL_PREFIX=${CODEBASE}/external/opencv \
	      -D BUILD_NEW_PYTHON_SUPPORT=OFF .
    fi
    make
    make install
    cd ..
fi

# zlib
if [ ! -e zlib ] && [ "$1" == "zlib" ]; then
    VERSION="1.2.8"
    wget -c http://zlib.net/zlib-${VERSION}.tar.gz || exit 1
    tar zxvf zlib-${VERSION}.tar.gz
    ln -s zlib-${VERSION} ${CODEBASE}/external/zlib
    cd zlib
    setenv CC gcc
    ./configure -t --shared
    make
    cd ..
fi

# lua
if [ ! -e lua ] && [ "$1" == "lua" ]; then
    VERSION="5.2.4"
    wget -c http://www.lua.org/ftp/lua-${VERSION}.tar.gz || exit 1
    tar xzvf lua-${VERSION}.tar.gz
    cd lua-${VERSION}
    if [ "$OS" == "Darwin" ]; then
        make macosx
    else
        make linux
    fi
    make install INSTALL_TOP=${CODEBASE}/external/lua
fi
