import itertools
import numpy as np
import tensorflow as tf

n=3
r = 2

def combination_ones(n,r):
    l = range(n)
    a = list(itertools.combinations(l, r))
    b = np.zeros([len(a),n])
    for e,a_ in enumerate(a):
        for i in a_:
            b[e,i] = 1
    return b

a = ["str",2,tf.placeholder("float32"),3,4,5]
print(np.array(a)[[0,2,3,4]])