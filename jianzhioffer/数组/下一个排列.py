# coding=utf-8
# 下一个排列
'''
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

#源于离散数学及其应用的算法：（以3 4 5 2 1 为例）
#从后往前寻找第一次出现的正序对：（找到 4,5）
#之后因为从5 开始都是逆序，所以把他们反转就是正序：3 4 1 2 5
#之后4 的位置应该是：在它之后的，比他大的最小值（5）
#交换这两个值：得到 3 5 1 2 4
# 对于初始即为逆序的序列，将在反转步骤直接完成
'''

class Solution(object):
    def nextPermutation(self, nums):
        if not nums or len(nums) < 2:
            return
        l = len(nums)
        i = l-1
        # 从后往前,寻找第一次出现的正序对  54 这种
        while i>=0 and nums[i-1] >= nums[i]:
            i -= 1
        a,b = i, l-1
        while a<b:  # 将已经扫描过的降序对变成升序对
            nums[a],nums[b] = nums[b],nums[a]
            a += 1
            b -= 1
        j = i - 1   # 这个是刚找到的那个正序对中，靠前的那个位置  也即例子中的4
        # 它的值应该是其后的最小的那个比它大的数
        for k in range(i, l):
            if nums[k] > nums[j]:
                nums[j], nums[k] = nums[k], nums[j]
                break
        return nums


s = Solution()
res = s.nextPermutation([5,4,3,2,1])
print res