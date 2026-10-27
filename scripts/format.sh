yapf -vv --style google -i t10n/*.py
yapf -vv --style google -i t10n/dataset/*.py
yapf -vv --style google -i exp/*/*.py
yapf -vv --style google -i exp/*/*/*.py
yapf -vv --style google -i test/*.py

clang-format --style=file -i csrc/*/*.h csrc/*/*.cc
clang-format --style=file -i csrc/*/*.cuh csrc/*/*.cu

python3 scripts/cpplint.py csrc/*/*.h csrc/*/*.cc
python3 scripts/cpplint.py csrc/*/*.cuh csrc/*/*.cu
