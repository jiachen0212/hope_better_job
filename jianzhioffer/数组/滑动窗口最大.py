# codin=utf-8
# 滑动窗口最大值
# maxpooling实现

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        win, res = [], []   # win存放没过期的，最大值的，index
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i - k:  # win[0] <= i - k 证明窗已经划滑过了
                win.pop(0)     # pop(0)把最早的index剔除
            while win and nums[win[-1]] <= v:
                win.pop()
            win.append(i)  # 把大的值的index加进来
            if i >= k - 1:
                res.append(nums[win[0]])  # 注意append的是win[0]
        return res