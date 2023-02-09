# coding=utf-8
# O(n)实现  遍历两遍数组即可

class Solution():
    def isEqual_3(self,arr):
        if not arr or len(arr)<3:
            return False
        l = len(arr)
        curSum = [0]*l  # 存放每一个至当前index位置的sum
        curSum[0] = arr[0]
        for i in range(1, l):
            curSum[i] = curSum[i-1]+arr[i]
        tmpSum = curSum[-1] / 3
        if tmpSum*3<curSum[-1]:
            return False    # 证明和是不被3整除的 可以直接false
        # 再开始遍历数组，开始寻找切分点
        for i in range(l):
            if curSum[i] == tmpSum:  # 找到第一个三分点
                if l-1-i<2:  # 后面需要有2段 长度<2肯定不行
                    return False
            if curSum[i] == 2*tmpSum:
                if l-1-i < 1:
                    return False
                else:
                    return True
        return False

s = Solution()
res = s.isEqual_3([3,1,2,1,1,1])
print(res)