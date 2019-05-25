FROM ubuntu:18.04

# Ubuntu setup
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Setup environment variables for Docker image

# setup home dir
ENV HOME /root
# env settings for OpenMC
ENV FC=/usr/bin/mpif90 CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx \
    PATH=$HOME/opt/openmc/bin:$HOME/opt/NJOY2016/build:$PATH \
    LD_LIBRARY_PATH=$HOME/opt/OPENMC/lib:$LD_LIBRARY_PATH \
    OPENMC_CROSS_SECTIONS=$HOME/nndc_hdf5/cross_sections.xml \
    OPENMC_ENDF_DATA=$HOME/endf-b-vii.1

# Install dependencies from Debian package manager
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common  build-essential && \
    apt-get install -y python3-setuptools \
                       python3-pip \
                       python3-setuptools \
                       python3-dev \
                       libpython3-dev \
                       python3-nose \
                       python3-matplotlib \
                       python3-tables \
                       python3-scipy \
                       python3-jinja2 && \
    apt-get install -y wget git vim && \
    apt-get install -y gfortran g++ cmake && \
    apt-get install -y mpich libmpich-dev && \
    apt-get install -y libhdf5-serial-dev libhdf5-mpich-dev hdf5-tools && \
    apt-get install -y imagemagick && \
    apt-get install -y libblas-dev liblapack-dev && \
    apt-get autoremove


# set default python to python3
#RUN echo 'alias python=python3' >> $HOME/.bashrc
#RUN echo 'alias pip=pip3' >> $HOME/.bashrc
#RUN /bin/bash -c "source $HOME/.bashrc"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 

# Update system-provided pip
RUN pip3 install --upgrade pip
RUN pip3 install --force-reinstall \
            sphinx \
            cloud_sptheme \
            prettytable \
            sphinxcontrib_bibtex \
            numpydoc \
            nbconvert \
            numpy \
            nose \
            cython \
            future \
            scipy

# Clone and install NJOY2016
RUN git clone https://github.com/njoy/NJOY2016 $HOME/opt/NJOY2016 && \
    cd $HOME/opt/NJOY2016 && \
    mkdir build && cd build && \
    cmake -Dstatic=on .. && make 2>/dev/null && make install

# Clone and install MOAB5.1
RUN mkdir -p $HOME/opt/MOAB5.1 && cd $HOME/opt/MOAB5.1 && \
    git clone https://bitbucket.org/fathomteam/moab && \
    cd moab && \
    git checkout -b Version5.1.0 origin/Version5.1.0 && \
    cd .. && \
    mkdir build && cd build && \
    # build/install static lib
    cmake ../moab -DCMAKE_INSTALL_PREFIX=$HOME/opt/MOAB5.1 \
              -DENABLE_HDF5=ON \
              #-DENABLE_PYMOAB=ON \
              -DBUILD_SHARED_LIBS=OFF  && \
    make -j 4 && \
    make install && \
    # build/install shared lib
    cmake ../moab -DCMAKE_INSTALL_PREFIX=$HOME/opt/MOAB5.1 \
              -DENABLE_HDF5=ON \
              #-DENABLE_PYMOAB=ON \
              -DBUILD_SHARED_LIBS=ON  && \
    make -j 4 && \
    make install && \
    cd .. && \
    rm -rf ./build
# put MOAB on the path
ENV LD_LIBRARY_PATH $HOME/opt/MOAB5.1/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH $HOME/opt/MOAB5.1/lib:$LIBRARY_PATH
ENV PYTHONPATH=$HOME/opt/MOAB5.1/lib/python3.6/site-packages/

# Clone and install DAGMC
RUN mkdir -p $HOME/opt/DAGMC  && cd $HOME/opt/DAGMC && \
    git  clone https://github.com/zxkjack123/DAGMC.git && \
    cd $HOME/opt/DAGMC/DAGMC && \
    git checkout develop && \
    git remote add upstream https://github.com/svalinn/DAGMC.git && \
    git pull upstream develop && \
    cd .. && mkdir bld && cd bld && \
    cmake ../DAGMC -DMOAB_DIR=$HOME/opt/MOAB5.1 \
                   -DCMAKE_INSTALL_PREFIX=$HOME/opt/DAGMC \
                   -DBUILD_TALLY=ON  && \
    make -j 4 && \
    make install && \
    cd .. && \
    rm -rf ./build
# add DAGMC lib into LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $HOME/opt/DAGMC/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH $HOME/opt/DAGMC/lib:$LIBRARY_PATH

# Clone and install OpenMC
RUN mkdir -p $HOME/opt/OPENMC &&  cd $HOME/opt/OPENMC  && \
    git clone https://github.com/zxkjack123/openmc.git && \
    cd openmc && \
    git remote add upstream https://github.com/openmc-dev/openmc.git && \
    git checkout develop && git pull upstream develop && \
    cd $HOME/opt/OPENMC/ && mkdir build && cd build && \
    cmake ../openmc \
        -Doptimize=on \
        -DHDF5_PREFER_PARALLEL=on \
        -DCMAKE_INSTALL_PREFIX=$HOME/opt/OPENMC \
        -Ddebug=ON \
        -Ddagmc=ON \
        -DDAGMC_ROOT=$HOME/opt/DAGMC && \
    make && make install && \
    cd ../openmc && pip install -e .[test]

ENV PATH=$HOME/opt/OPENMC/bin:$PATH
ENV LD_LIBRARY_PATH=$HOME/opt/OPENMC/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$HOME/opt/OPENMC/lib:$LIBRARY_PATH

# Download cross sections (NNDC and WMP) and ENDF data needed by test suite
RUN $HOME/opt/OPENMC/openmc/tools/ci/download-xs.sh
