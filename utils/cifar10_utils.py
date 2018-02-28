"""
 data visulize
"""
import numpy as np
import matplotlib.pyplot as plt

def try_asnumpy(data):
    try:
        data = data.asnumpy() # if is <class 'mxnet.ndarray.ndarray.NDArray'>
    except BaseException:
        pass
    return data

def show_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]),
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5):
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
            if N * i + j < images.shape[0]:
                image = (images[N * i + j] / 255).clip(0, 1)
                figs[i][j].imshow(image)
                
                figs[i][j].axes.get_xaxis().set_visible(False)
                figs[i][j].axes.get_yaxis().set_visible(False)
                
                if labels is not None:
                    label = labels[N * i + j]
                    for l in label:
                        if l[0] < 0: continue
                        l[1], l[2], l[3], l[4] = l[1] * w, l[2] * h, l[3] * w, l[4] * h
                        rect = box_to_rect(l[1:5], color, linewidth)
                        figs[i][j].add_patch(rect)
                        if show_text:
                            figs[i][j].text(l[1], l[2], str(int(l[0])), 
                                            bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))
            else:
                figs[i][j].set_visible(False)
    plt.show()

def text_label_to_int(text_labels):
    d = {'horse':7, 'automobile':1, 'deer':4, 'dog':5, 'frog':6, 'cat':3, 'truck':9, 'ship':8, 'bird':2, 'airplane':0}
    return [d[text_label] for text_label in text_labels]

def get_text_labels(label, text_labels=None):
    if text_labels is None:
        text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in label]
