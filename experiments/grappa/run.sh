#python3 build_chunks.py
#python3 t10nrun.py --host_file hosts_docker --tag_id gindbg --task train --train_num_proc 1

#python3 t10nrun.py --host_file hosts --tag_id partition --task partition

python3 t10nrun.py --host_file hosts_h100 --tag_id gindbg --task train --train_num_proc 2
