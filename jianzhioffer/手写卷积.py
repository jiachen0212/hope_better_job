# coding=utf-8
# 手写卷积

# import numpy as np
# input = np.array([[40,24,135],[200,239,238],[90,34,94]])
# kernel = np.array([[0.0,0.6],[0.1,0.3]])

# def my_conv(input,kernel):
#     output_size = (len(input)-len(kernel)+1)
#     res = np.zeros([output_size,output_size],np.float32)
#     for i in range(len(res)):
#         for j in range(len(res)):
#             res[i][j] = compute_conv(input,kernel,i,j)
#     return res

# def compute_conv(input,kernel,i,j):
#     res = 0
#     for kk in range(kernel.shape[0]):
#         for k in range(kernel.shape[1]):
#             res +=input[i+kk][j+k] * kernel[kk][k]  #这句是关键代码，实现了两个矩阵的点乘操作
#     return res
# print(my_conv(input,kernel))


Input = [[40,24,135],[200,239,238],[90,34,94]]
kernel = [[0.0,0.6],[0.1,0.3]]

def my_conv(input,kernel):
    output_size = (len(input)-len(kernel)+1)
    res = [[0]*output_size for i in range(output_size)]
    for i in range(len(res)):
        for j in range(len(res)):
            res[i][j] = int(compute_conv(input,kernel,i,j))
    return res

def compute_conv(input,kernel,i,j):
    res = 0
    for kk in range(len(kernel)):
        for k in range(len(kernel[0])):
            res +=int(input[i+kk][j+k]) * float(kernel[kk][k])
    return res

print my_conv(Input, kernel)