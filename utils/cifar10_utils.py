"""
 data visulize
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def try_asnumpy(data):
    try:
        data = data.asnumpy() # if is <class 'mxnet.ndarray.ndarray.NDArray'>
    except BaseException:
        pass
    return data

def show_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]),
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5, xlabels=None, ylabels=None):
    """
    advise to set dpi to 120
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
    
    images: numpy images type, shape is (n, 3, h, w), or (n, 2, h, w), pixel value range 0~255, float type
    labels: boxes, shape is (n, m, 5), m is number of box, 5 means every box is [label_id, xmin, ymin, xmax, ymax]
    rgb_mean: if images has sub rgb_mean, shuold specified.
    MN: is subplot's row and col, defalut is (-1, 5), -1 mean row is adaptive, and col is 5
    """
    images = try_asnumpy(images)
    labels = try_asnumpy(labels)
    
    if MN is None:
        M, N = (images.shape[0] + 4) / 5, 5
    else:
        M, N = MN
    _, figs = plt.subplots(M, N, figsize=figsize)
    
    images = (images.transpose((0, 2, 3, 1)) * std) + rgb_mean
    h, w = images.shape[1], images.shape[2]
    for i in range(M):
        for j in range(N):
            fig = figs[i][j] if M > 1 else figs[j]
            if N * i + j < images.shape[0]:
                image = (images[N * i + j] / 255).clip(0, 1)
                fig.imshow(image)
                
                if xlabels is not None: 
                    fig.set_xlabel(xlabels[N * i + j], fontsize=fontsize)
                if ylabels is not None: 
                    fig.set_ylabel(ylabels[N * i + j], fontsize=fontsize)
                
                fig.set_xticks([])
                fig.set_yticks([])
#                 fig.axes.get_xaxis().set_visible(False)
#                 fig.axes.get_yaxis().set_visible(False)
                
                if labels is not None:
                    label = labels[N * i + j]
                    for l in label:
                        if l[0] < 0: continue
                        l[1], l[2], l[3], l[4] = l[1] * w, l[2] * h, l[3] * w, l[4] * h
                        rect = box_to_rect(l[1:5], color, linewidth)
                        fig.add_patch(rect)
                        if show_text:
                            fig.text(l[1], l[2], str(int(l[0])), 
                                            bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))
            else:
                fig.set_visible(False)
    plt.show()

def text_label_to_int(text_labels):
    d = {'horse':7, 'automobile':1, 'deer':4, 'dog':5, 'frog':6, 'cat':3, 'truck':9, 'ship':8, 'bird':2, 'airplane':0}
    return [d[text_label] for text_label in text_labels]

def get_text_labels(label, text_labels=None):
    if text_labels is None:
        text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in label]


"""
    log parser
"""
import matplotlib.pyplot as plt
import numpy as np

def show_log(filename, ignore_sharp=True):
    for line in open(filename).readlines():
        if ignore_sharp and len(line.strip()) > 0 and line.strip()[0] == "#":
            continue
        print line,

def parse_sharp(line):
    def find_tuple(line):
        idx1 = line.find("(")
        idx2 = line[idx1+1:].find(")") + idx1+1
        w_strs = line[idx1+1:idx2].split(",")[:-1]
        w = [0] * len(w_strs)
        for i in range(len(w_strs)):
            w[i] = float(w_strs[i])
        return w, idx1, idx2
    
    line  = line[1:]
    w, _, idx2 = find_tuple(line)
    g, _, _ = find_tuple(line[idx2+1:])
    return w, g
    

def parse_log(log_file, begin_line=0, to_float_k=["train_acc", "valid_acc", "loss"]):
    obj = {'weight':[], 'grad':[]}
    with open(log_file) as f:
        lines = f.readlines()[begin_line:]
        for line in lines:
            line = line.strip()
            if line[0:1] == "#": 
                w, g = parse_sharp(line)
                if len(w) > 0 and len(g) > 0:
                    obj['weight'].append(w)
                    obj['grad'].append(g)
                continue
            vs = []
            for d in line.split(","):
                d = d.strip()
                i = d.rfind(' ')
                k = d[:i].strip()
                v = d[i+1:].strip()
                obj[k] = obj.get(k, [])
                obj[k].append(v)
        if len(lines) > 0:
            for k in to_float_k:
                if obj.has_key(k):
                    obj[k] = to_float(obj[k])
    return obj

def to_float(str_list):
    fa = []
    for s in str_list:
        fa.append(float(s))
    return fa

def plot(data, key, x_range=None):
    x_range = (0, len(data[key])) if x_range is None else x_range
    if x_range[1] == -1:  x_range = (x_range[0], len(data[key]))
    plt.plot(range(*x_range), data[key][x_range[0]:x_range[1]], label=key)
    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel(key)
    
dataset = {}
data_init = {"train_acc": []}

def update(log_file, x_range=None):
    dataset[log_file] = dataset.get(log_file, {"train_acc": []})
    data = dataset[log_file]
    
    def get_begin_line(log_file, data):
        epochs = len(data["train_acc"])
        if epochs == 0: return 0, 0
        l = 0
        with open(log_file) as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if line[0:1] != "#":
                    l += 1
                if l >= epochs:
                    return i+1, l
    
    # update date
    begin_line, _  = get_begin_line(log_file, data)
    _data = parse_log(log_file, begin_line=begin_line)
    for k in _data:
        data[k] = data.get(k, [])
        data[k].extend(_data[k])
    if not data.has_key("valid_acc") or not data.has_key("train_acc") or not data.has_key("loss"):
        return
    
    # plot
    plt.figure(figsize=(12, 4))   # (w, h)
    plt.subplot(1, 2, 1)
    plot(data, 'train_acc', x_range)
    plot(data, 'valid_acc', x_range)
    #plt.show()
    plt.subplot(1, 2, 2)
    plot(data, 'loss', x_range)
    plt.show()
    print "lr", sorted(to_float(set(data['lr'])), reverse=True)
    
    # weight and grad
    data = dataset[log_file]["weight"]
    if len(data) == 0: return
    data = np.array(data)
    grad = dataset[log_file]["grad"]
    if len(grad) == 0: return
    grad = np.array(grad)
    print np.mean(data, axis=0), np.var(data, axis=0)
    print np.mean(grad, axis=0), np.var(grad, axis=0)