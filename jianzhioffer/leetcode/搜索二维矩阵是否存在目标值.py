# coding=utf-8
# 搜索二维矩阵


# leetcode代码
# 转为一维数组做，index 转换 [index/col][index%col]
def sm1(matrix, target):
    if not matrix or not matrix[0]:
        return False
    row = len(matrix)
    col = len(matrix)
    l, r = 0, row*col - 1
    while r>= l:
        mid = (l + r) / 2
        if matrix[mid/col][mid%col] == target:
            return True
        elif matrix[mid/col][mid%col] > target:
            r = mid - 1   # 这个r l 的更新要很小心
        else:
            l = mid + 1
    return False



# 类似剑指offer的，比较每行的最末元素，可以剔除这一行
def sm(matrix, target):
    if not matrix or not matrix[0]:
        return False
    row = len(matrix)
    col = len(matrix[0])
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == target:
                return True
            elif matrix[i][col - 1] < target:
                break   # 可以跳出这一行了
    return False


matrix =  [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
print(sm(matrix, target))