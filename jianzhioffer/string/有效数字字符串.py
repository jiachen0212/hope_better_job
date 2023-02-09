#coding=utf-8
##############  牛客 ac 版
class Solution:
    # s字符串
    def isNumber(self, s):
        if len(s) == 0:
            return False
        point_done = False
        e_done = False
        if s[0] in '+-':
            s = s[1:]
        for i in range(len(s)):
            if s[i] in 'eE':
                s = s[i + 1 :]
                e_done = True
                break
            elif s[i] == '.':
                s = s[i + 1 :]
                point_done = True
                break
            elif s[i] < '0' or s[i] > '9':   # 这句用elif not s[i].isdigit():也行
                return False
        if point_done == True:
            for i in range(len(s)):
                if s[i] in 'Ee':
                    s = s[i + 1 :]
                    e_Done = True
                    break
                elif s[i] < '0' or s[i] > '9':
                    return False
        if e_done == True:
            if len(s) == 0:
                return False
            if s[0]  in '+-':
                s = s[1:]
            if len(s) < 1:
                return False
            for x in s:
                if x < '0' or x > '9':
                    return False
        return True



# 本地ok 但是牛客没过...
# 判断是都字符串可以表示一个数值
# 可以的形式: A.Be|EC / .Be|EC  AC有正负号,B无符号
def isNumeric(s):
    isAllowDot = True  # 表示还没出现小数点.
    isAllowE = True    # 表示还没出现e/E   e/E/. 都是只能出现一次的..
    for i in range(len(s)):
        if i == 0 and s[i] == 'e': # e前面必须有数值
            return False
        if s[i] in "+-" and (i == 0 or s[i-1] in "eE") and i < len(s)-1:
            continue
        elif isAllowDot and s[i] == ".": # 有小数点,则其前面的A可以是0,但后面的B必须右值
            isAllowDot = False
            if i >= len(s)-1 or s[i+1] not in "0123456789": # 这就是后面没值. 即使是小数点后面有e,e的前面也得有系数啊..
                return False
        elif isAllowE and s[i] in "Ee":
            isAllowDot = False # 这个有必要! 使得e后面的小数点.会return False
            isAllowE = False  # 这里是需要的,因为e的指数不可为小数.
            if i >= len(s)-1 or s[i+1] not in "0123456789+-":
                return False
        elif s[i] not in "0123456789":  # 排除第一个字符是e的情况.
            return False
    return True

res = isNumeric('+1.2e+12')
print res