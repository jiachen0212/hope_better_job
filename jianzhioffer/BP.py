#coding=utf-8
import forwardcalu as forward
import numpy as np

def conv_backward_naive(dout, cache):
    dx, dw, db = None, None, None
    N, F, H1, W1 = dout.shape
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    HH = w.shape[2]
    WW = w.shape[3]
    S = conv_param['stride']
    P = conv_param['pad']
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    x_pad = np.pad(x, [(0,0), (0,0), (P,P), (P,P)], 'constant')
    dx_pad = np.pad(dx, [(0,0), (0,0), (P,P), (P,P)], 'constant')
    db = np.sum(dout, axis=(0,2,3))

    for n in xrange(N):
      for i in xrange(H1):
        for j in xrange(W1):
        # Window we want to apply the respective f th filter over (C, HH, WW)
          x_window = x_pad[n, :, i * S : i * S + HH, j * S : j * S + WW]
          for f in xrange(F):
            dw[f] += x_window * dout[n, f, i, j] #F,C,HH,WW
            #C,HH,WW
            dx_pad[n, :, i * S : i * S + HH, j * S : j * S + WW] += w[f] * dout[n, f, i, j]

    dx = dx_pad[:, :, P:P+H, P:P+W]
    return dx, dw, db

x_shape = (2, 3, 4, 4)
w_shape = (2, 3, 3, 3)
x = np.ones(x_shape)
w = np.ones(w_shape)
b = np.array([1,2])
conv_param = {'stride': 1, 'pad': 0}
Ho = (x_shape[3]+2*conv_param['pad']-w_shape[3])/conv_param['stride']+1
Wo = Ho
dout = np.ones((x_shape[0], w_shape[0], Ho, Wo))
out, cache = forward.conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

print "out shape",out.shape
print "dw=========================="
print dw
print "dx=========================="
print dx
print "db=========================="
print db