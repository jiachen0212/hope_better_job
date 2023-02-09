# coding=utf-8
'''
在一个N*N的数组中寻找所有横，竖，左上到右下，右上到左下，
四种方向的直线连续D个数字的和里面最大的值
'''

s=raw_input().split()
print(s)
N=int(s[0])
D=int(s[1])
print(N, D)
a=[]
for i in range(N):
    s=raw_input().split()
    temp=[]
    for x in s:
        temp.append(int(x))
    a.append(temp)
res=0
for i in range(N):
    for j in range(N):
        temp=[0,0,0,0]   # 四个方向的和
        if j<=N-D:  # 横
            temp[0]=sum(a[i][j:j+D])
        if i<=N-D:  # 纵向
            for k in range(D):
                temp[1]+=a[i+k][j]
        if i<=N-D and j<=N-D:  # 斜向上
            for k in range(D):
                temp[2]+=a[i+k][j+k]
        if i<=N-D and j>=D-1: # 斜向下
            for k in range(D):
                temp[3]+=a[i+k][j-k]
        if res<max(temp):
            res=max(temp)
print(res)