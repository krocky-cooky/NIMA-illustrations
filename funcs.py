import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import os,sys

def img_check(id,path,height = 1):
    h = 6.4*height
    im = Image.open(os.path.join(path,str(id) + '.jpg'))
    im_list = np.asarray(im)
    w = h*im_list.shape[1]/float(im_list.shape[0])
    print(im_list.shape)
    fig = plt.figure(figsize = (h,w))
    ax = fig.add_subplot(111)
    ax.imshow(im_list)
    plt.show()
    return im_list