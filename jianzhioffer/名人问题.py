# coding=utf-8
# https://www.cnblogs.com/rgvb178/p/10117404.html

# 名人问题：这个人谁都不认识，但是大家都认识他
'''
如果a认识b，则a不会是名人；如果a不认识b，则b不会是名人。因此每询问一次a是否认识b，都可以排除掉一个人，所以在O(n)时间内就可以排除掉n-1个人。  最后还要检查确认，是否其他人都认识这个人，以及这个人都不认识其他人。
'''

# The knows API is already defined for you.
# @param a, person a
# @param b, person b
# @return a boolean, whether a knows b
# def knows(a, b):


# 一共需要询问3(n-1)次——时间复杂度为O(n)
class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return -1
        curr_stay = 0
        # 询问n-1次，即可找到这个名人
        for i in range(1, n):
            if knows(curr_stay, i):
                curr_stay = i

        # check 环节
        for i in range(0, n):
            if i == curr_stay:
                continue
            # 这个名人认识别人？那不对 return -1
            if knows(curr_stay, i):
                return -1
            # 有人不认识这个名人？那也不对 return -1
            if not knows(i, curr_stay):
                return -1
        return curr_stay

s = Solution()
res = s.findCelebrity(100)
print(res)
