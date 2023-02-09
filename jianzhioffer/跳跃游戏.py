# coding=utf-8
# 跳到数组尾部需要几步？
class Solution(object):
    def canJump(self, nums):
        l = len(nums)
        if l==1:
            return 0

        q = []
        res = 0
        visited = [False for i in range(l)]
        # index表示nums中的index
        q.append(0)
        visited[0] = True

        while q:
            for j in range(len(q)):
                node = q.pop(0)
                for i in range(nums[node], 0, -1): # 从最大开始找有助于加快速度
                # 最大可以跳nums[node]步长.
                    new_index = node+i
                    if new_index >= l-1:
                        return res + 1
                    if not visited[new_index]:
                        visited[new_index] = True
                        q.append(new_index)
            res += 1

s = Solution()
res = s.canJump([1,2,1,1,4])
print res




# 是否可以跳到尾部
# 从后往前遍历
class Solution(object):
    def canJump(self, nums):
        if not nums or len(nums) <=1:
            return True
        lenth = 1
        l = len(nums)
        for i in range(l-2,0,-1):  #1~l-2
            if nums[i] < lenth:
                lenth += 1
            else:
                lenth = 1
        return lenth <= nums[0]