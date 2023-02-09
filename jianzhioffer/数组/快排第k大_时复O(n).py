# coding=utf-8
# 时间复杂度O(n)
# 排序后的nums并不是完全有序，只需找到第k大就好

# 这个其实是第k小 且从0开始计数
# 因为我们把小的放前面，大的放后面
def change(nums, left, right):
    key = nums[left]
    while left < right:
        while (left < right) and (key <= nums[right]):
            right -= 1
        if left < right:  # 跳出while,即key>nums[right]
            nums[left] = nums[right]  # 把这个小了的right换到前面去
        while (left < right) and (nums[left] <= key):
            left += 1
        if left < right:  # 跳出while,即nums[left]>key
            nums[right] = nums[left]  # 把这个大的left换到后面去
    nums[left] = key
    return left

def _quick_sort(nums, left, right, k):
    if left < right:
        key_ind = change(nums, left, right)
        if key_ind < k:
            _quick_sort(nums, key_ind+1, right, k)
        elif key_ind > k:
            _quick_sort(nums, left, key_ind-1, k)

def main(nums, k):
    _quick_sort(nums, 0, len(nums)-1, k)
    return nums[k]

if __name__ == '__main__':
    import random
    # a = [random.randint(0, 100) for x in range(10)]
    a = [3,2,1,5,6,4]
    print(a)
    print(main(a, 2))
    print(sorted(a))




# leetcode  第k大
# class Solution(object):
#     def _quick_sort(self, s, l, r, k):
#         if l < r:
#             key_ind = self.change(s, l, r)
#             if k < key_ind:
#                 self._quick_sort(s, l, key_ind-1, k)
#             elif k > key_ind:
#                 self._quick_sort(s, key_ind+1, r, k)

#     def change(self, s, l, r):
#         key = s[r]
#         while l < r:
#             while l < r and key<= s[l]:
#                 l += 1
#             if l < r:
#                 s[r] = s[l]

#             while l < r and s[r] <= key:
#                 r -= 1
#             if l < r:
#                 s[l] = s[r]
#         s[r] = key
#         return r

#     def findKthLargest(self, s, k):
#         if not k or not s or k > len(s)-1:
#             return None
#         self._quick_sort(s, 0, len(s)-1, k)
#         return s[k]

# s = Solution()
# res = s.findKthLargest([3,2,1,5,6,4], k-1)
# print res