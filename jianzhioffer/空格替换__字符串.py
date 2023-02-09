# 牛客ac

# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, strs):
        len1 = len(strs)
        if len1 == 0:
            return ''

        # 获得空格的个数
        block = 0
        for i in range(len1):
            if strs[i] == ' ':
                block += 1

        len2 = len1 + 2 * block
        newstr = [0] * len2  # 预存新的str

        # p1 p2 分别指向空格替换前后，str的末尾
        # 从后往前，使得只需要遍历一次str 时间复杂度O(0)
        p1 = len1 - 1
        p2 = len2 - 1
        while p1 >= 0 and p2 >= p1:
            if strs[p1] != ' ':
                newstr[p2] = strs[p1]
                p2 -=1
            else:  # 遇上空格
                newstr[p2] = '0'
                newstr[p2 - 1] = '2'
                newstr[p2 - 2] = '%'
                p2 -= 3
            p1 -= 1
        return ''.join(newstr)