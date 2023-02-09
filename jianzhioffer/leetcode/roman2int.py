# coding=utf-8
# roman2int
class Solution(object):
    def romanToInt(self, s):
        result=0
        dic={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        for i in s[-1::-1]:
            symbol=1
            if (i in ['I','X','C']) and result>=5*dic[i]:
                symbol=-1
            result+=dic[i]*symbol
        return result

s = Solution()
print(s.romanToInt('IV'))
