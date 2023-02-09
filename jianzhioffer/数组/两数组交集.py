# coding=utf-8
# 两数组交集
# leetcode
# https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/comments/
# 时复：O(m+n)   空复是O(m)

class Solution(object):
    def intersect(self, nums1, nums2):
        res = []
        if not nums1 or not nums2:
            return res
        dict1={}.fromkeys(nums1,0)
        for num in nums1:
            dict1[num] += 1

        for num in nums2:
            if num in dict1.keys() and dict1[num]:  # and dict1[num] 为了确保dict1中的value还>0
                res.append(num)
                dict1[num] -= 1  # 这里dict1自减很重要
        return res


s = Solution()
res = s.intersect([4,9,5],[9,4,9,8,4])
print(res)









# 抖机灵做法....
class Solution(object):
    def intersect(self, nums1, nums2):
        res = []
        nums = set(nums1) & set(nums2)
        for num in nums:
            res += [num]*min(nums1.count(num), nums2.count(num))
        return res