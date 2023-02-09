#coding=utf-8
# 调整数组顺序使得奇数在偶数前面
# 调整数组顺序使得被3整除的在不被整除的后面  使用func()做判断
# 技巧是：使用2个指针定位前后两个元素,一奇一偶的话就调换他们的位置  使用func1()做判断
# 精简技巧是代码大框架不变,对于变换位置的条件单独用函数拎出来,实现一个代码可多功能.

# 实现是否偶数的判断
def func(a, ind):
    if a[ind] & 0x1 == 0:
        return True

# 实现是否被3整除的判断
def func1(a, ind):
    if a[ind] % 3 != 0:
        return True

def fun(a):
    # 首尾指针
    ind1 = 0
    ind2 = len(a) - 1
    while ind2 > ind1:
        if func1(a, ind1): # 第一个指向了偶数
            if not func1(a, ind2): # 并且这时候第二个指向的是奇数,则需要交换两个元素
                temp = a[ind2]
                a[ind2] = a[ind1]
                a[ind1] = temp
            else: # 第二个指针指向的是偶数,则需要前移第二个指针直到找到奇数.
                ind2 -= 1
        else: # 第一个指针指向的是奇数,直接后移第一个指针直到找打接下来的一个偶数.
            ind1 += 1
    print a
    return a

a = [1, 3, 3, 4, 4, 6, 7, 8, 9, 2, 4, 12]
fun(a)





