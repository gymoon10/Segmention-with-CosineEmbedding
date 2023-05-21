from datasets.CityscapesDataset import CityscapesDataset
from datasets.KomatsunaDataset import KomatsunaDataset
from datasets.jabcho_indiv import jabcho_indiv

from datasets.CitrusDataset import CitrusDataset
from datasets.CornDataset import CornDataset

def get_dataset(name, dataset_opts):
    if name == "cityscapes": 
        return CityscapesDataset(**dataset_opts)
    elif name == 'cvppp':
        return CVPPPDataset(**dataset_opts)
    elif name == 'cvppp2':
        return CVPPPDataset2(**dataset_opts)
    elif name == 'cvppp2_indiv':
        return CVPPPDataset2_indiv(**dataset_opts)
    elif name == 'komatsuna':
        return KomatsunaDataset(**dataset_opts)
    elif name == 'jabcho_indiv':
        return jabcho_indiv(**dataset_opts)
    elif name == 'Citrus':
        return CitrusDataset(**dataset_opts)
    elif name == 'Corn':
        return CornDataset(**dataset_opts)
    elif name == 'Corn0':
        return CornDataset0(**dataset_opts)

    else:
        raise RuntimeError("Dataset {} not available".format(name))