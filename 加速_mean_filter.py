def MeanFilter(im, r):
    H,W = im.shape 
    res = np.zeros((H,W))
    # 行维度上, 由上至下逐渐加法, 做行上的积分
    integralImagex = np.zeros((H+1,W))
    for i in range(H):    
        integralImagex[i+1,:] = integralImagex[i,:]+im[i,:]
    # 以r为单位, 对应做减法, 起到下面多一行, 上面就减一行的效果 
    # 得到的mid, 就是r行上, 各行的积分值(完成了行上加法)
    mid = integralImagex[r:]-integralImagex[:-r]
    # /r 就是在行维度上做mean处理   
    mid = mid / r  

    # 行上做左右padding
    padding = r - 1 
    leftPadding = (r-1)//2 
    rightPadding = padding - leftPadding

    # 基本后第i行值padding
    left = integralImagex[r-leftPadding:r]
    # 原im的值padding
    right = integralImagex[-1:] - integralImagex[-r+1:-r+1+rightPadding]

    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(-1,1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(-1,1)
    left /= leftNorm
    right /= rightNorm
    im1 = np.concatenate((left,mid,right))

    # 相同方式处理列
    integralImagey = np.zeros((H,W+1))
    res = np.zeros((H,W))
    for i in range(W):    
        integralImagey[:,i+1] = integralImagey[:,i]+im1[:,i]
    mid = integralImagey[:,r:]-integralImagey[:,:-r] 
    mid = mid / r 
    left = integralImagey[:,r-leftPadding:r]
    right = integralImagey[:,-1:] - integralImagey[:,-r+1:-r+1+rightPadding]
    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(1,-1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(1,-1)
    left /= leftNorm
    right /= rightNorm
    im2 = np.concatenate((left,mid,right),axis=1)
    
    return im2