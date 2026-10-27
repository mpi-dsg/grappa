from t10n.dataset.dgl_to_t10n import load_as_dgl_g
from t10n.dataset.dgl_to_t10n import augmentation_dgl_g
from t10n.dataset.dgl_to_t10n import to_t10n_from_dgl_g
from t10n.dataset.dgl_to_t10n import to_t10n_from_downloaded_ds

DS_ORI = '$DATA_PATH/ds_ori/'
DS_ORI = '$DATA_PATH/dgl/'
DS_T10N = '$DATA_PATH/t10n/'

#DS_ORI = '/data/ds_ori/'
#DS_T10N = '/data/t10n/'

for dataset_name in ['mag']:
#    #for dataset_name in ['ogbpr', 'ogbar', 'cora', 'reddit']:
    graph = load_as_dgl_g(dest_path=DS_ORI, dataset_name=dataset_name)
    graph = augmentation_dgl_g(graph)
    print(graph.edges()[0].shape)
    exit(0)
    to_t10n_from_dgl_g(dest_path=DS_T10N,
                       dataset_name=dataset_name,
                       dgl_g=graph)
exit(0)
dataset_name = 'igb260m'
src_path = DS_ORI
to_t10n_from_downloaded_ds(dest_path=DS_T10N,
                           dataset_name=dataset_name,
                           src_path=src_path)
