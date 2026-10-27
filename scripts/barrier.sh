for i in `seq 0 15`;
do 
    #sed -i 's/300/3600 * 5/g' /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/agent/server/api.py;
    #sed -i 's/3600/7200/g' /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/agent/server/api.py
    #sed -i 's/300/7200*5/g' /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/utils/store.py
    #ssh t10n_cu118_w$i sed -i 's/300/7200*5/g' /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py
    ssh t10n_cu118_w$i cat /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/agent/server/api.py | grep 7200
    ssh t10n_cu118_w$i cat /usr/local/lib/python3.10/dist-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py | grep 7200
done
