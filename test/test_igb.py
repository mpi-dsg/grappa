from t10n.dataset.dgl_to_t10n import load_as_dgl_g
from t10n.dataset.meta import igb_meta

DS_ORI = '$DATA_PATH/ds_ori/'
#DS_ORI = '/data/ds_ori/'
#DS_T10N = '/data/t10n/'

meta = igb_meta()
SRC_PATH = f"{DS_ORI}/{meta.igb_path}"
dgl_g = load_as_dgl_g(SRC_PATH, "igb260m")
