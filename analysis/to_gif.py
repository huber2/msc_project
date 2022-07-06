from distutils.log import Log
from multiprocessing import allow_connection_pickling
from os.path import dirname, abspath
from PIL import Image
import numpy as np


DIR_PATH = dirname(abspath(__file__)) + '/../'

arrays = np.load('data/video3.npz', allow_pickle=True)['arr_0']
arrays = np.array(arrays * 255, dtype=np.uint8)

imgs = [Image.fromarray(a) for a in arrays]

imgs[0].save('data/video3.gif', save_all=True, append_images=imgs[1:])