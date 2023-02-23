def convolve2d_vector(arr, kernel, stride=1, padding='same'):
    h, w, channel = arr.shape[0],arr.shape[1],arr.shape[2]
    k, n = kernel.shape[0], kernel.shape[2]
    r = int(k/2)
    #重新排列kernel为左乘矩阵，通道channel前置以便利用高维数组的矩阵乘法
    matrix_l = kernel.reshape((1,k*k,n)).transpose((2,0,1))
    padding_arr = np.zeros([h+k-1,w+k-1,channel])
    padding_arr[r:h+r,r:w+r] = arr
    #重新排列image为右乘矩阵，通道channel前置
    matrix_r = np.zeros((channel,k*k,h*w))
    for i in range(r,h+r,stride):
        for j in range(r,w+r,stride): 
            roi = padding_arr[i-r:i+r+1,j-r:j+r+1].reshape((k*k,1,channel)).transpose((2,0,1))
            matrix_r[:,:,(i-r)*w+j-r:(i-r)*w+j-r+1] = roi[:,::-1,:]        
    result = np.matmul(matrix_l, matrix_r)
    out = result.reshape((channel,h,w)).transpose((1,2,0))
    return out[::stride,::stride]

if __name__=='__main__':
    import numpy as np 
    import time 
    N=1000
    A = np.arange(1,N**2+1).reshape((N,N,1))
    kernel = np.arange(3**2).reshape((3,3,1))/45
    A3 = np.concatenate((A, 2*A, 3*A), axis=-1)
    k3 = np.concatenate((kernel, kernel, kernel), axis=-1)
    print(A3.shape, k3.shape)

    t1 = time.time()
    B2 = convolve2d_vector(A3,k3,stride=2).astype(np.int)
    t2 = time.time()
    print(t2-t1)
    print(B2.shape)
