# coding=utf-8
# 数组，奇数左边, 偶数右边  且保持奇数奇数之间  偶数偶数之间的相对位置不变


######### 牛客 ac
def reOrderArray(array):
        odd,even=[],[]   # 需要申请额外的空间
        for i in array:
            odd.append(i) if i%2==1 else even.append(i)
        return odd+even
a = [1,2,3,5,6,7]
print(reOrderArray(a))


##### 也ac了，把奇偶数改成字母大小写而已.....
# def reOrderArray(array):
#         small,big=[],[]   # 需要申请额外的空间
#         for i in array:
#             big.append(i) if i.isupper() else small.append(i)
#         return ''.join(small + big)
# a = 'shFYGiKNgui'
# print(reOrderArray(a))

