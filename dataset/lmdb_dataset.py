from posixpath import join
import torch.utils.data as data
import numpy as np
import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, noise_model=None, size=None, repeat=1, ratio_used_list=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            length = txn.stat()['entries']

        self.length = size or length
        if ratio_used_list is not None:
            idx_used = []
            for i in range(self.length):
                if int(self.meta[i][3]) in ratio_used_list:
                    idx_used.append(i)
            print(f"Ratio used to train: {ratio_used_list}")
            print(f"Used pairs: {len(idx_used)} out of {self.length}")
        self.repeat = repeat
        self.meta = pickle.load(open(join(db_path, 'meta_info.pkl'), 'rb'))
        self.shape = self.meta['shape']
        self.dtype = self.meta['dtype']
        self.noise_model = noise_model

    def __getitem__(self, index):
        env = self.env
        index = index % self.length
        
        with env.begin(write=False) as txn:
            raw_data = txn.get('{:08}'.format(index).encode('ascii'))

        flat_x = np.frombuffer(raw_data, self.dtype)
        x = flat_x.reshape(*self.shape)
        
        if self.dtype == np.uint16:
            x = np.clip(x / 65535, 0, 1).astype(np.float32)
        if len(self.meta[index]) == 2:
            wb, color_matrix = self.meta[index]
            ratio, K = -1, -1
        else:
            wb, color_matrix, ISO, ratio = self.meta[index]
            if self.noise_model is not None:
                K = self.noise_model.ISO_to_K(ISO)
            else:
                K = -1

        return x, {"ratio": ratio, "K": K} # None: noise params

    def __len__(self):
        return int(self.length * self.repeat)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
