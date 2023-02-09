# coding=utf-8
# 字符串分割
'''
如“abc,de,fg” 输出 ["abc","de","fg"]
'''


def fun(s):
    ll = len(s)
    res = []
    p,tmp = 0,0
    while p < ll:
        if s[p] != ',':
            p+=1
        else:
            res.append(s[tmp:p])
            tmp = p+1
            p = tmp

        # 这两行特别注意！因为最后的那个fg后面没有,使得它加入res
        # 所以需要手动加一下！
        # 很容易遗漏啊！！！
        if p == ll-1:
            res.append(s[tmp:p+1])
    return res

s = "abce,fg"
print(fun(s))

