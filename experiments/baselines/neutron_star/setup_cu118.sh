apt install -y unzip

mkdir -p build && cd build

git clone https://github.com/iDC-NEU/NeutronStarLite.git
cd NeutronStarLite && git reset --hard 18f74cd7340b110a724ead7ee77c1bec16c9adf3
git am ../../0001-*.patch
git am ../../0002-*.patch
git am ../../0003-*.patch
git am ../../0004-*.patch
git am ../../0005-*.patch
git am ../../0006-*.patch
git am ../../0007-*.patch

## torch 1.7.1, newer version has error
wget -O libtorch_171_cu110.zip https://download.pytorch.org/libtorch/cu110/libtorch-shared-with-deps-1.7.1%2Bcu110.zip
unzip libtorch_171_cu110.zip



#export CUDA_HOME=/usr/local/cuda-11.3/
export CUDA_HOME=/usr/local/cuda-11.8/
#export MPI_HOME=/usr/local/mpi/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$MPI_HOME/bin:$CUDA_HOME/bin:$PATH
export PATH=/usr/bin:$CUDA_HOME/bin:$PATH

mkdir build && cd build && cmake ..
make -j4

cd ../
cp -r ../../t10n_baseline ../
cd t10n_baseline && ls
