# coding=utf-8
'''
确定第一位数：a = k/(n-1)!
确定第二位数：更新k,k=k-a*(n-1)!
            b = k/(n-2)!
            ...

'''
class Solution(object):
    def getPermutation(self, n, k):
        tmp = [1]
        for i in range(1, n+1):
            tmp.append(tmp[-1]*i)

        data = '0123456789'   # 这个要理解 为什么要用一下它
        # 因为除以了n-1!证明这一为肯定是'满'的吧？ 其实还是不是很理解....
        res = ''
        while k>=1 and n:
            pos = k/tmp[n-1]
            if k % tmp[n-1] != 0:
                pos += 1
            k -= (pos-1)*tmp[n-1]
            res += data[pos]
            data = data[:pos] + data[pos+1:]
            n -= 1
        return res


s = Solution()
res = s.getPermutation(4,15)
print res




