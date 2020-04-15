import numpy as np
import sys
from scipy.misc import imsave

num = int(sys.argv[1])
a = np.load('X.npy').T
i = (a[num]*255).reshape(40,40,3)
print(i.shape)
imsave('x-view.jpg', i)
