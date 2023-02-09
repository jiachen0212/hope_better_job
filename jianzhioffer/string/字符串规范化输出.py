# coding=utf-8
# 字符串规范化输出
# 小大小  小大大  大大小
# 所以考虑连续三个str 中间的必须是大  第一个只要是小 第三个随变
# 第一个是大的话 第三个必须是小
# 就ok了



def fun(s):
    l = len(s)
    res = s[0].lower()
    for i in range(1, l-1):
        if s[i].isupper() and ((s[i-1].islower() or (s[i-1].isupper() and s[i+1].islower()))):
            res += '_'
        res += s[i].lower()
        # if i == l - 2 and s[i].isupper():
            # if s[-1].islower():
            #     res += '_'
    res += s[-1].lower()
    return res


ans = fun('LeetHTTPBackCHENJ')
print ans