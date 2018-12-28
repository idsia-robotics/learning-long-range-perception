import numpy as np

coords = [(i + 100, j) for i in range(-400, 400 + 1, 50) for j in range(-400, 400 + 1, 50)]
coord_strs = [str(x) + '_' + str(y) for x, y in coords]
