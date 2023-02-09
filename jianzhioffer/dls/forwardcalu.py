import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    out = None
    N,C,H,W = x.shape     # N C W H
    F,_,HH,WW = w.shape   # F C WW HH
    S = conv_param['stride']
    P = conv_param['pad']

    #output size
    Ho = 1 + (H + 2 * P - HH) / S
    Wo = 1 + (W + 2 * P - WW) / S
    x_pad = np.zeros((N,C,H+2*P,W+2*P))
    x_pad[:,:,P:P+H,P:P+W]=x
    out = np.zeros((N,F,Ho,Wo))

    for f in xrange(F):
      for i in xrange(Ho):
        for j in xrange(Wo):
          out[:,f,i,j] = np.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], axis=(1, 2, 3))
      out[:,f,:,:]+=b[f]
    cache = (x, w, b, conv_param)
    return out, cache

x_shape = (2, 3, 4, 4) # N C W H
w_shape = (2, 3, 3, 3) # F C WW HH
x = np.ones(x_shape)
w = np.ones(w_shape)
b = np.array([1, 2])
conv_param = {'stride': 1, 'pad': 0}
out, _ = conv_forward_naive(x, w, b, conv_param)
print out
# print out.shape

