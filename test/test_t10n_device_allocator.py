from t10n.dataloader import T10nDeviceAllocator

a = T10nDeviceAllocator(2)

subprocess.call("gcc -o numabench numa_bench.c -lnuma -fopenmp", shell=True)

a.set_omp_dev_assignments(4, 0)
for i in range(5):
    subprocess.call("./numabench", shell=True)

a.set_omp_dev_assignments(4, 1)
for i in range(5):
    subprocess.call("./numabench", shell=True)
