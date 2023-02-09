# coding=utf-8
# https://blog.csdn.net/znzxc/article/details/82826873
# 商汤code题，MxN 正/长方形个数

'''
坐标为（x,y）,把这个点当做长方形的左上角定点，则此长方形的宽度变化范围是（1，n-x），
高度变化范围是（1,y）,对应的长方形个数是（n-x）*y，
再减去高和宽相等的情况，即（n-x）*y - min（n-x,y）
'''

# 有多少个长方形/正方形
def many_cza(m, n):
    c = 0
    z = 0
    areas = set()
    for x in range(m):
        for y in range(n):
            width = m - x
            height = n - y
            c += width * height - min(width, height)
            z += min(width, height)
            areas.add(width*height)
    return c, z, len(areas)

c, z, area = many_cza(4,3)
print(c,z,area)

