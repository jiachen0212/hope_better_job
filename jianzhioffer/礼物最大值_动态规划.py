#coding=utf-8
# 礼物的最大值,也即动态规划问题.
# 矩阵的左上角出发,每次只能往右和下走,每个格子对应一个数值,使得最终路径值最大.

class Solution:
    def getmaxValue(self, values, rows, cols):
        if not values or rows <= 0 or cols <= 0:
            return 0
        # 用于存放中间数值的临时数组
        temp = [0] * cols  # 只需要考虑一维就够.
        for i in range(rows):
            for j in range(cols):
                left = 0  # 向右走的,所以上一步是left
                up = 0    # 向下走的,所以上一步是up
                if i > 0:
                    up = temp[j]
                if j > 0:
                    left = temp[j-1]
                temp[j] = max(up, left) + values[i*rows+j]
        return temp[-1]
s = Solution()
a = s.getmaxValue([1,10,3,8,12,2,9,6,5,7,4,11,3,7,16,5],4,4)
print a