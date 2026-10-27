 for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "ps -aux | grep t10n_train | awk '{print \$2}' | xargs kill -9"; done
 for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "pkill -f python3"; done
