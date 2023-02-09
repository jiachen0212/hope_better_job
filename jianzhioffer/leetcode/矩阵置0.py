# coding=utf-8
# 矩阵置0
# 某个位置(i,j)上出现0，则将i行和j列均置0


# 空间复杂度O(m+n)
def setZeros(matrix):
    row = len(matrix)
    col = len(matrix[0])
    flg = [0] * (row + col)
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                flg[i] = 1
                flg[row+j] = 1

    for i in range(row):
        if flg[i] == 1:
            matrix.pop(i)
            matrix.insert(i, [0] * col)
    for hang in matrix:
        for j in range(col):
            if flg[row+j] == 1:
                hang.pop(j)
                hang.insert(j, 0)
    return matrix




# 空间复杂度O(1)
# 把[i,j]是否0的信息都放到第一行和第一列中去体现
def setZeros1(matrix):
    row = len(matrix)
    col = len(matrix[0])
    fr, fc = 0, 0
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                matrix[i][0] = 0   # 好了第i行要准备清零0，flag信息放下[i][0]位置处
                # 这个信息是存放在原matrix中的，所以不需要额外的内存占用
                matrix[0][j] = 0   # 同上，j列准备清0吧

                # 给第一行和第一列设置flag
                if i == 0:
                    fr = 1
                if j == 0:
                    fc = 1

    # 注意这里没对第一行和第一列处理，因为现在它放flag的，不方便现在处理它
    for i in range(1, row):
        for j in range(1, col):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # 现在来处理第一行和第一列
    if fr == 1:
        for i in range(col):
            matrix[0][i] = 0
    if fc == 1:
        for j in range(row):
            matrix[j][0] = 0
    return matrix



matr = [
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
# res = setZeros(matr)
# print(res)

res1 = setZeros1(matr)
print(res1)

