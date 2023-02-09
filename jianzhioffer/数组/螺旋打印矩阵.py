# coding=utf-8
# leetcode ac
class Solution:
    def spiralOrder(self, matrix):
        # matrix = matrix.tolist()  # matrix转list
        res = []
        while matrix:
            # matrix.pop(0) 直接就是第一行
            res += matrix.pop(0)   # 打印第一行，最上边 左至右直接添加

            # 这三行实现第二条边上至下打印
            if matrix and matrix[0]: # matrix[0]应该是保证列边还有元素吧
                for row in matrix:
                    res.append(row.pop()) # .pop()函数默认删除最末尾元素，正好是我们要的竖边元素                print(res, 'res1')

            if matrix:
                res += matrix.pop()[::-1]   # matrix.pop()表示取出最后一行, [::-1]表示逆序从后往前打印

            if matrix and matrix[0]:   # 第四条边下至上打印
                for row in matrix[::-1]:   # matrix[::-1] 这里也要逆序请注意，因为是从下到上！！！
                    res.append(row.pop(0)) # pop(0) 表示删除的是第一个元素
        return res


b = [[1, 3, 5, 7], [12, 13, 14, 9], [11, 16, 15, 2], [10, 8, 6, 4]]
s = Solution()
res = s.spiralOrder(b)
print(res)





import numpy as np
# 方法一 顺时针一圈圈打印矩阵的元素,每一圈的打印分四步:左-右 上-下 右-左 下-上
# 方法二 先打印一行然后就删掉这一行(list中的pop可实现),完了以后把矩阵转置并按列倒序reverse()一下!这个操作是精髓的地方. 转置+倒序后再接着打印首行并删这行,再转置..迭代重复至打完即可...

# 方法一:
# 顺时针一圈圈打印矩阵的元素
print '#######方法一#######'
def printCircle(a, col, row, start):
    endX = col - 1- start
    endY = row - 1 - start

    # 从左到右边打印一行. 因为range取不到最后一个数,所以这里进行了endX+1
    for i in range(start, endX + 1):
        print a[start][i]
    # 打印从上到下的一列
    if start < endY:
        for i in range(start + 1, endY + 1):  # start + 1是因为刚才那里已经把第一行的最末一个元素打印了,所以接着打印的话要下一行然后再打.
            # endY + 1的原因和上面endX一样,因为range取不到最后一个数
            print a[i][endX]
    # 从右到左打印
    if start < endX and start < endY:  # start<endX是因为打印第一段的时候并没有考虑endX和start的值关系. 然后start<endY是因为这一圈不至于只有一行
        for i in range(endX - 1, start - 1, -1):  # endX-1是因为上一步已经把圈的右下角元素打印了.. start-1是因为range取不到这个值.因为是逆序,所以是减1
            print a[endY][i]
    # 从上到下打印
    if start < endX and start < endY -1:  # 因为上一步和第一步都打印了,所以行之间必须2行才行..
        for i in range(endY - 1, start, -1):  # endY-1是因为endY行上一步已经打印了, 直到start+1行,因为圈的左上角第一步就打印了.但是因为range取不到最后的值,所以start+1-1=start
            print a[i][start]
def printmatix1(a):
    col, row = a.shape[:2]
    if col <= 0 or row <= 0:
        return True
    start = 0
    while 2 * start < col and 2 * start < row:
        printCircle(a, col, row, start)
        start += 1

a = np.array([[1, 2, 3, 4, 5], [16, 17, 18, 19, 6], [15, 24, 25, 20, 7], [14, 23, 22, 21, 8], [13, 12, 11, 10, 9]])
b = np.array([[1, 3, 5, 7], [12, 13, 14, 9], [11, 16, 15, 2], [10, 8, 6, 4]])
# printmatix1(b)


# print '#######方法二#######'
print b
# 方法二 先打印一行然后就删掉这一行(list中的pop可实现),完了以后把矩阵转置并按列倒序reverse()一下!这个操作是精髓的地方. 转置+倒序后再接着打印首行并删这行,再转置..迭代重复至打完即可...
class Solution:
    def printMatrix2(self, matrix):
        matrix = matrix.tolist()  # matrix转list
        while matrix:  # 判断list是否为空
            result = matrix.pop(0)  # pop()函数的功能 取出list的一行放入result,并在原list中删除这行
            # 打印这一行..
            for i, num in enumerate(result):
                print num
            # if not matrix[0]:
            #     break    # 矩阵空了...
            matrix = self.turn(matrix)  # 对矩阵进行旋转
            # return result
    def turn(self, matrix):
        matrix = np.array(matrix)
        matrix = matrix.T  # 矩阵转置
        matrix = matrix.tolist()
        matrix.reverse()  # 将转置后的矩阵每一列首尾调换,得到我们需要的魔方矩阵.
        # mat = np.array(matrix)  # 可视化检查下转置T+reverse后的矩阵是否正确
        # print mat
        return matrix

# s = Solution()
# s.printMatrix2(b)

