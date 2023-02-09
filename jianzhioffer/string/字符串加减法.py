# coding=utf-8
# 字符串加减法

def calue(s):
    if  s == '':
        return 0
    s = s.strip()  # 去掉字符串中的空格
    sub_str = ''
    res = 0

    if s[0] not in '+-':   # 首字符不是+-的话，手动加一下
        s = '+' + s
    ll = len(s)
    p1, p2 = 0, 1
    while p2 < ll-1:
        if s[p1] in '+-':
            sub_str += s[p1]
            while s[p2].isdigit():
                sub_str += s[p2]
                p2 += 1
            res += int(sub_str)
            sub_str = ''
            p1 = p2
            p2 += 1
    res += int(s[-2:])
    return res


# s = '1+11+345-13+44-7+90-294-3-3+4'
# ans = calue(s)
# print(ans)






class Solution:
    def calculate(self, s):
        stack = []
        sign = 1
        num = 0
        res = 0
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == "+":
                res += sign * num
                num = 0
                sign = 1
            elif c == "-":
                res += sign * num
                num = 0
                sign = -1 # sign会作用在下一个num  sign*num
            elif c == "(":
                stack.append(res) # 先压入之前的res
                stack.append(sign) # 再压入括号内鹅符号
                sign = 1
                res = 0  # 进入括号，之前的res已经存进stack了，所以把res清0
            elif c == ")":
                res += sign * num  # 计算括号内的这个res值
                num = 0
                # 后入先出  所以是先出符号  再出以前的res值
                res = stack.pop() * res + stack.pop()
        res += sign * num
        return res