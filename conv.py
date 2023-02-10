# coding=utf-8
# 实现nchw的卷积计算
import numpy as np

def conv_naive(x, out_c, ksize, padding=0, stride=1):
    # x = [b, h, w, in_c]
    b, h, w, in_c = x.shape
    kernel = np.random.rand(ksize, ksize, in_c, out_c)
    if padding > 0:
        pad_x = np.zeros((b, h+2*padding, w+2*padding, in_c))
        pad_x[:,padding:-padding,padding:-padding,:] = x

    out_h = (h+2*padding-ksize)//stride+1
    out_w = (w+2*padding-ksize)//stride+1
    out = np.zeros((b, out_h, out_w, out_c))

    for i in range(out_h):
        for j in range(out_w):
            # 每次stride步长移动x
            roi_x = pad_x[:,i*stride:i*stride+ksize,j*stride:j*stride+ksize,:]
            # roi_x = [b, ksize, ksize, in_c] -> [b, ksize, ksize, in_c, out_c]
            # kernel = [ksize, ksize, in_c, out_c]
            # conv = [b, ksize, ksize, in_c, out_c] -> [b, 1, 1, out_c]
            conv = np.tile(np.expand_dims(roi_x, -1), (1,1,1,1,out_c))*kernel
            out[:,i,j,:] = np.squeeze(np.sum(conv, axis=(1,2,3), keepdims=True), axis=3)
    return out

# 补充下2d卷积code
def conv2d(Input,kernel,padding=0,stride=2):
    h_in, w_in = Input.shape[:2]
    k = kernel.shape[0]   # 默认正方形kernel
    h_out, w_out = (h_in+2*padding-k)//stride+1, (w_in+2*padding-k)//stride+1
    res = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out): 
            # 注意别忘了input的ij移动要乘上stride!!!
            sub_input = Input[i*stride:i*stride+k, j*stride:j*stride+k]
            res[i][j] = np.sum(sub_input * kernel)
    return res
Input = np.array([[40,24,135,1],[200,239,238,1],[90,34,94,1], [1,2,3,4]])
kernel = np.array([[0.0,0.6],[0.1,0.3]])
padding, stride = 0, 2
out_put = conv2d(Input, kernel, padding=padding, stride=stride)
print('Conv2d:', out_put)


if __name__ == '__main__':
    input_ = np.random.rand(1,10,10,3)
    out = conv_naive(input_, out_c=15, ksize=3, padding=1, stride=2)
    print(out.shape)

