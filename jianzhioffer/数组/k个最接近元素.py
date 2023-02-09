# coding=utf-8
# 找到k个最接近的元素

class Solution(object):
    def findClosestElements(self, arr, k, x):
        ll = len(arr)
        l,r = 0,ll-k   # 注意这里r是ll-k
        while l <r:
            mid = (r+l)/2
            if (x-arr[mid] > arr[mid+k]-x):
            # 证明mid选小了 所以l可以后移
                l=mid+1
            else:
                r=mid
        return arr[l:l+k]

s = Solution()
print(s.findClosestElements([1,2,3,4,5], 4, 3))



