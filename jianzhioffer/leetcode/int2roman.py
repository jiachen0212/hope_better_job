# coding=utf-8
# Integer to Roman
class Solution(object):
    def intToRoman(self, num):
        unit = [1000,900,500,400,100,90,50,40,10,9,5,4,1]   # 由大到小排序就好～
        unit_str = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
        rstr = ''
        r = num
        for i in range(13):
            if r < unit[i]:
                continue
            d, r = divmod(r, unit[i])
            rstr = rstr+unit_str[i]*d
        return rstr
