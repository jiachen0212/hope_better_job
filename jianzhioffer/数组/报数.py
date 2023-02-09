# coding=utf-8
'''
1.     1
2.     11
3.     21
4.     1211
5.     111221

1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。

'''
# 没懂......

class Solution(object):
    def countAndSay(self, n):
        tmp, now = '1','1'
        for i in range(2,n+1):
            now = ''
            count = 1
            for j in range(1, len(tmp)):
                if tmp[j] == tmp[j-1]:
                    count += 1
                else:
                    now += str(count)
                    now += tmp[j-1]
                    count = 1
            now += str(count)
            now += tmp[-1]
            tmp = now
        return now

s = Solution()
res = s.countAndSay(4)
print res