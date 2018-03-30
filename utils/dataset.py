import os

import warnings
from mxnet import gluon, nd, image
class MultiFolderDataset(gluon.data.dataset.Dataset):
    """A dataset for loading ndarray files or image files stored in a folder structure like::

        roots[0]/car/0001.ndarray
        roots[0]/car/xxxa.ndarray
        roots[0]/car/yyyb.ndarray
        roots[0]/bus/123.ndarray
        roots[0]/bus/023.ndarray
        roots[0]/bus/wwww.ndarray
        
        roots[1]/car/0001.ndarray
        roots[1]/car/xxxa.ndarray
        roots[1]/car/yyyb.ndarray
        roots[1]/bus/123.ndarray
        roots[1]/bus/023.ndarray
        roots[1]/bus/wwww.ndarray

    Parameters
    ----------
    root : str
        Path to root directory.
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::

        transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all ndarrays in (filename, label) pairs.
    """
    def __init__(self, roots, flag=1, transform=None):
        self._roots = []
        for root in roots:
            self._roots.append(os.path.expanduser(root))
        self._flag = flag
        self._transform = transform
        self._exts = ['.ndarray', '.jpeg', '.jpg', '.png']
        self._label_dict = {}
        self.synsets = []
        self.items = []
        for root in self._roots:
            self._list_images(root)

    def _list_images(self, root):
        for folder in sorted(os.listdir(root)):
            if folder[0] == '.': continue
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
                
            if not self._label_dict.has_key(folder):
                self._label_dict[folder] = len(self.synsets)
                self.synsets.append(folder)
            label = self._label_dict[folder]
            
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        if (os.path.splitext(self.items[idx][0])[1]).lower() == '.ndarray':
            data = nd.load(self.items[idx][0])[0]
        else:
            data = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(data, label)
        return data, label

    def __len__(self):
        return len(self.items)
    
    
import warnings
from mxnet import gluon, nd
class NDArrayFolderDataset(gluon.data.dataset.Dataset):
    """A dataset for loading ndarray files stored in a folder structure like::

        root/car/0001.ndarray
        root/car/xxxa.ndarray
        root/car/yyyb.ndarray
        root/bus/123.ndarray
        root/bus/023.ndarray
        root/bus/wwww.ndarray

    Parameters
    ----------
    root : str
        Path to root directory.
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::

        transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all ndarrays in (filename, label) pairs.
    """
    def __init__(self, root, transform=None):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.ndarray']
        self._list_images(self._root)

    def _list_images(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        data = nd.load(self.items[idx][0])[0]
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(data, label)
        return data, label

    def __len__(self):
        return len(self.items)
    
class MyArrayDataset(gluon.data.dataset.Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    data_list : one or more dataset-like objects
        The data arrays.
        
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::

        transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, data_list, transform=None):
        assert len(data_list) > 0, "Needs at least 1 arrays"
        self._length = len(data_list[0])
        self._data = []
        for i, data in enumerate(data_list):
            assert len(data) == self._length, \
                "All arrays must have the same length; array[0] has length %d " \
                "while array[%d] has %d." % (self._length, i+1, len(data))
            if isinstance(data, nd.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            self._data.append(data)
        self._transform = transform

    def __getitem__(self, idx):
        if len(self._data) == 1:
            if self._transform is not None:
                return self._transform(self._data[0][idx])
            return self._data[0][idx]
        else:
            if self._transform is not None:
                return self._transform(*tuple(data[idx] for data in self._data))
            return tuple(data[idx] for data in self._data)

    def __len__(self):
        return self._length

class MyArrayDataset2(gluon.data.dataset.Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    data_list : one or more dataset-like objects, or list of dataset-like objects
        The data arrays.
        
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::

        transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def _cal_len(self, data):
        length = 0
        if isinstance(data, list):
            for i in range(len(data)): length += len(data[i])
        else: length = len(data)
        return length
    
    def __init__(self, data_list, transform=None):
        assert len(data_list) > 0, "Needs at least 1 arrays"
        self._length = self._cal_len(data_list[0])
        self._data = []
        for i, data in enumerate(data_list):
            assert self._cal_len(data) == self._length, \
                "All arrays must have the same length; array[0] has length %d " \
                "while array[%d] has %d." % (self._length, i+1, self._cal_len(data))
            if isinstance(data, nd.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            if isinstance(data, list):
                assert len(data) > 0, "Needs at least 1 arrays for data_list[%d]" % i
                for di in range(len(data)):
                    if isinstance(data[di], nd.NDArray) and len(data[di].shape) == 1:
                        data[di] = data[di].asnumpy()
            self._data.append(data)
        self._transform = transform
        
    def _get_data(self, data_id, idx):
        if isinstance(self._data[data_id], list):
            for i in range(len(self._data[data_id])):
                if idx < len(self._data[data_id][i]):
                    return self._data[data_id][i][idx]
                idx -= len(self._data[data_id][i])
        return self._data[data_id][idx]
    
    def __getitem__(self, idx):
        if len(self._data) == 1:
            if self._transform is not None:
                return self._transform(self._get_data(0, idx))
            return self._get_data(0, idx)
        else:
            if self._transform is not None:
                return self._transform(*tuple(self._get_data(i, idx) for i in range(len(self._data))))
            return tuple(self._get_data(i, idx) for i in range(len(self._data)))

    def __len__(self):
        return self._length
    