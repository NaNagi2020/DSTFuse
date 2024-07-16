import torch.utils.data as Data
import h5py
import numpy as np
import torch

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir'][key])
        VIS = np.array(h5f['vis'][key])
        h5f.close()
        return key, torch.Tensor(VIS).unsqueeze(0), torch.Tensor(IR).unsqueeze(0)
    
    def get_label(self):
        h5f = h5py.File(self.h5file_path, 'r')
        keys = list(h5f['label'].keys())
        # label = np.array(h5f['label'].values())
        label =  {key: h5f['label'][key][()] for key in keys}
        h5f.close()
        return label

