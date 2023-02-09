# coding=utf-8
################### leecode ac 版
# -*- coding:utf-8 -*-
# 这个版本，出现小数点或者字母类的字符的话，全部都return 0
class Solution(object):
    def myAtoi(self, str):
        temp_str = str.lstrip()   # 祛除首部的空格
        if temp_str and temp_str[0] in '+-':
            temp_num = temp_str[0]   # str中有正负号的话就用起来
            temp_str = temp_str[1:]
        else:
            temp_num = '+'    # 没有的话就直接给默认的+号
        for i in temp_str:
            if i.isdigit():
                temp_num += i
            else:
                break  # 就退出了，后面不管了
        if len(temp_num) >= 2:
            if int(float(float(temp_num))) >= 2**31:
                return 2**31-1
            elif int(float(float(temp_num))) <= -(2**31):
                return -(2**31)
            else:
                return int(float(float(temp_num)))
        else:
            return 0

ss = Solution()
ans = ss.StrToInt('4193 with words')
print(ans)


#### 这份代码是更鲁棒性的，允许string里有字母、小数点等字符，但是牛客没ac  可能越界啥的...
# string2int
class Solution:
    def myAtoi(self, string):
        temp_str = string.lstrip()   # 去除字符串首部的空格
        float_flag = True

        if temp_str and temp_str[0] in ['+', '-']:
            temp_num = temp_str[0]
            temp_str = temp_str[1:]  # 第一个字符是符号,则数字往后看
        else:
            temp_num = '+'

        for i in temp_str:
            if i.isdigit():   # 是数字
                temp_num += i
            elif i == '.' and float_flag:
                temp_num += i
                float_flag = False  # 遇到小数点可以终止了
            else:
                break

        if len(temp_num) >= 2:
            if float(temp_num) > (2**31 - 1):
                return 2**31 - 1
            elif float(temp_num) < (-2**31):
                return -2**31
            else:
                return int(float(temp_num))
        else:
            return 0

s = Solution()
res = s.myAtoi('-123.7')
# print(res)

