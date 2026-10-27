mkdir -p build/ && cd build/

git clone https://github.com/GATECH-EIC/BNS-GCN.git
cd BNS-GCN
git reset --hard b4a48bdaf25212bfddce46ec84af0ca2e351be7a

git am ../../0001-BNS_GCN-*.patch
git am ../../0002-BNS_GCN-*.patch
git am ../../0003-BNS_GCN-*.patch
