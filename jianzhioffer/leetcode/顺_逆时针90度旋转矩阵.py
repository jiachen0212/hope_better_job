# coding=utf-8



# 顺时针
def rotate1(A):
    A[:] = zip(*A[::-1])
    return A


# 逆时针矩阵旋转
def roate2(A):
     A[:] = zip(*A)[::-1]
     return zip(*A)[::-1]


a = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]
b = rotate1(a)
print(b)

c = roate2(a)
print(c)

