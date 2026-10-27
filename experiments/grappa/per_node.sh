#python3 build_chunks.py
#python3 t10nrun.py --host_file hosts_docker --tag_id per_node --task train --train_num_proc 1 --train_script train.sh.template.dbg

# brain 4 gpu per node
python3 t10nrun.py --host_file hosts_docker --tag_id per_node --task train --train_num_proc 1 --train_script train.sh.template.dbg
