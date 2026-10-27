from t10n.dataset.dgl_to_t10n import load_as_dgl_g as t10n_load_dgl_g

DATA_ROOT = "$DATA_PATH/"
ori_path = f"{DATA_ROOT}/ds_ori"
dataset_name = "ogbpr"

dgl_g = t10n_load_dgl_g(ori_path, dataset_name)

print(dgl_g)
