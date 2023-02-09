# coding=utf-8
# 两玩家，一枚两枚，硬币取胜问题

# n 为硬币个数，一次可以取1个或者2个，谁取走最后的硬币谁胜，问首家胜利的可能序列

# 首家一定要胜的话，取决于硬币的个数。

def fun(n):
    f = [False for i in range(n)]
    f[1] = True
    f[2] = True  # f[0]位置我就不管，硬币为1个或者2个的话，必定是首先家赢的！
    f[3] = False
    f[4] = True
    for i in range(5, n):
        # (f[i-2] and f[i-3]) 是对方拿一个，我拿一个或者2个  1+1 1+2
        # (f[i-3] and f[i-4]) 是对方拿两个，我拿一个或者2个   2+1 2+2
        f[i] = (f[i-2] and f[i-3]) or (f[i-3] and f[i-4])
    return(f)

ans = fun(10)
print(ans)
