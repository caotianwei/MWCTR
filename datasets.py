"""
doc
"""
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class HistDataset(Dataset):
    __doc__ = "Dataset with user's behavioral history."

    def __init__(self, data_list, pad_size, pad_value):
        """
        Args:
            data (list): [[user, [item_1, item_2, ...], item, label], ...]
            pad_value (int): Padding the historical list with this.
            pad_size (int): The length of the padded historical list.
        """
        self.pad_size = pad_size
        self.pad_value = pad_value
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        user, hist_item, neg_hist_item, item, label = self.data_list[idx]
        hist_size = len(hist_item)
        hist_item += [self.pad_value] * (self.pad_size - hist_size)
        neg_hist_item += [self.pad_value] * (self.pad_size - hist_size)
        return (user, item, hist_item, neg_hist_item, label)


class Data:
    __doc__ = 'Manage experimental data.'

    def __init__(self, data_path, hp=False):
        """
        Args:
            data_path (str): Load data from this path on the disk.
        """
        with open(data_path, 'rb') as (file):
            train_set = pickle.load(file)
            val_set = pickle.load(file)
            test_set = pickle.load(file)
            self.cate_list = list(pickle.load(file))
            self.user_count, self.item_count, self.cate_count, self.max_len = pickle.load(
                file)

        if hp:
            train_set = train_set[:len(train_set) // hp]
            val_set = val_set[:len(val_set) // hp]
            test_set = test_set[:len(test_set) // hp]
        self.train_set = HistDataset(train_set, self.max_len, self.item_count)
        self.val_set = HistDataset(val_set, self.max_len, self.item_count)
        self.test_set = HistDataset(test_set, self.max_len, self.item_count)

    def produce_loader(self, data_flag, batch_size, shuffle=True):
        """
        Produce a data loader.
        Args:
            data_flag (str): 'train_set', 'val_set' or 'test_set'
            batch_size (int): The mini-batch size.
            shuffle (bool): Weather shuffle the dataset.
        """
        data_set = getattr(self, data_flag)
        assert isinstance(data_set, HistDataset)
        return DataLoader(data_set, batch_size, shuffle,
                          collate_fn=(Data._wrap_batch), pin_memory=True)

    @staticmethod
    def _wrap_batch(batch):
        """
        Convert the mini-batch data as tensors.
        """
        *batch_x, batch_y = list(zip(*batch))
        # batch_x = list(map(torch.cuda.LongTensor, batch_x))
        # batch_y = torch.cuda.FloatTensor(batch_y).unsqueeze(1)
        batch_x = list(map(torch.LongTensor, batch_x))
        batch_y = torch.FloatTensor(batch_y).unsqueeze(1)
        return (batch_x, batch_y)


if __name__ == '__main__':
    data = Data('amazon_dataset_val_5.pkl')
    data_loader = data.produce_loader('train_set', 2, True)
