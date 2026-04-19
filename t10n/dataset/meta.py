#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

class reddit_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 232965
        # number of edges
        self.n_e = 114848857  #114615892 w/o loop
        self.origin_n_e = 114615892
        # dim of feature
        self.n_dim = 602
        # number of labels
        self.n_label = 41
        # dataset name
        self.name = "reddit"
        self.origin_name = "reddit"
        # path to find original data
        self.origin_path = "reddit"


class ogbpr_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 2449029
        # number of edges
        self.n_e = 126167053  # 61859140 is the number before preprocessing
        self.origin_n_e = 61859140
        # dim of feature
        self.n_dim = 100
        # number of labels
        self.n_label = 47
        # dataset name
        self.name = "ogbpr"
        self.origin_name = "ogbn-products"
        # path to find original data
        self.origin_path = "ogbn_products/raw"


class ogbar_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 169343
        # number of edges
        self.n_e = 1335586  # 1166243 w/o self-loop
        self.origin_n_e = 1166243
        # dim of feature
        self.n_dim = 128
        # number of labels
        self.n_label = 40
        # dataset name
        self.name = "ogbar"
        self.origin_name = "ogbn-arxiv"
        # path to find original data
        self.origin_path = "ogbn_arxiv/raw"


class ogbpa_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 111059956
        # number of edges
        self.n_e = 1726745828  # 1615685872 is the number before preprocessing
        self.origin_n_e = 1615685872
        # dim of feature
        self.n_dim = 128
        # number of labels
        self.n_label = 172
        # dataset name
        self.name = "ogbpa"
        self.origin_name = "ogbn-papers100M"
        # path to find original data
        self.origin_path = "ogbn_papers100M/raw"


class cora_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 2708
        # number of edges
        self.n_e = 13264  # 10556 w/o self-loop
        self.origin_n_e = 10556
        # dim of feature
        self.n_dim = 1433
        # number of labels
        self.n_label = 7
        # dataset name
        self.name = "cora"
        self.origin_name = "cora"
        # path to find original data
        self.origin_path = "cora_v2"


class igb_meta:

    def __init__(self) -> None:
        # number of nodes
        self.n_n = 269346174
        # number of edges
        self.n_e = 3727095830
        # 3996442004  # with self-loop
        self.origin_n_e = 3727095830
        # dim of feature
        self.n_dim = 1024
        # number of labels
        self.n_label = 19
        # dataset name
        self.name = "igb260m"
        self.origin_name = "igb260m"
        # path to find original data
        self.origin_path = "igb/IGB-Datasets/igb/igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/"
        self.igb_path = "igb/IGB-Datasets/igb/igb-public.s3.us-east-2.amazonaws.com/IGBH/"

class rmat_meta:

    def __init__(self, scale=26) -> None:
        # number of nodes
        self.n_n = 2 ** scale
        # number of edges
        self.n_e = 2 ** (scale+4)
        # 3996442004  # with self-loop
        self.origin_n_e = self.n_e
        # dim of feature
        self.n_dim = 128
        # number of labels
        self.n_label = 19
        # dataset name
        self.name = f"rmat_{scale}"
        self.origin_name = self.name
        # path to find original data
        self.origin_path = "rmat"
        self.igb_path = "rmat"

class mag_meta:
    def __init__(self) -> None:
        self.name = "mag"
        self.origin_name = "mag240m"
        # number of nodes
        self.n_n = 244160499
        # number of edges
        self.n_e = 1728364232
        # 3996442004  # with self-loop
        self.origin_n_e = 1728364232
        # dim of feature
        self.n_dim = 244160499
        # number of labels
        self.n_label = 349

def name_to_meta(name: str):
    supported_ds = ['reddit', 'ogbpr', 'ogbar', 'ogbpa', 'cora', 'mag', 'igb260m']
    assert name in supported_ds, f"dataset {name} in not supported"
    if name == 'reddit':
        meta_o = reddit_meta()
    if name == 'ogbpr':
        meta_o = ogbpr_meta()
    if name == 'ogbar':
        meta_o = ogbar_meta()
    if name == 'ogbpa':
        meta_o = ogbpa_meta()
    if name == 'cora':
        meta_o = cora_meta()
    if name == 'igb260m':
        meta_o = igb_meta()
    if name == 'mag':
        meta_o = mag_meta()
    if 'rmat' in name:
        return rmat_meta()
    return meta_o


def t10n_chunk_path(dataset_name, part_method, n_part):
    return f"chunking_{dataset_name}_{part_method}_{n_part:03}"
