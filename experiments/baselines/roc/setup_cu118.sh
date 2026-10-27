mkdir -p ./build
mkdir -p /build && cd /build

git config --global --add safe.directory /workspace/t10n/exp/roc/build/ROC/

git clone --recurse-submodules https://github.com/jiazhihao/ROC.git
#git submodule update --init --recursive
cd ROC/ && git reset --hard 1f5810811599a863b6305ba674d0f97cdac1e734
git am /workspace/t10n/exp/roc/0001-ROC-*.patch

cd legion/
git am /workspace/t10n/exp/roc/0001-legion-*.patch

cd ..
# not needed in cu118
rm -r cub/

make -j16

cp gnn /workspace/t10n/exp/roc/t10n_baseline/

cd /workspace/t10n/exp/roc/t10n_baseline/ && ls
