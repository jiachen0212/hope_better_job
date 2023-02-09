# coding=utf-8

# 左边维护一堆小，右边维护一堆大!
def leftmin_rightmax(num, left, right):
    if (len(left)+len(right)) % 2 == 0: # 偶数个
        if left and num < max(left):
            left.append(num)
            num = max(left)
            left.remove(num)
        right.append(num)  # 理论上第一个数是加在right边
    else:
        if right and num > min(right):
            right.append(num)
            num = min(right)
            right.remove(num)
        left.append(num)
    return left, right


def main(arrs):
    left, right = [],[]
    for num in arrs:
        left, right = leftmin_rightmax(num, left, right)
    if len(arrs)%2 == 0:
        res = (max(left) + min(right))/2
    else:
        res = min(right)  # 理论上右边会多一个数
    return res

# in_datas = raw_input()
# arrs = in_datas.split(' ')
ans = main([1,2,13,-5,6,-4])
print(ans)