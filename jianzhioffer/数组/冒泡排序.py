# coding=utf-8
# 冒泡排序
# 时间复杂度  平均/最坏：O(n^2) 最好：O(n)  稳定

def maopao_sort(s):
    ll = len(s)
    for i in range(ll):  # 这个是依次把第i大的数放到末尾
        # 一直把大的值放到最后去，所以第二层的循环是ll-i-1 因为后面
        # 的数已经是有序的了，所以无需仔遍历
        for j in range(ll-i-1):
            if s[j] > s[j+1]:
                s[j], s[j+1] = s[j+1], s[j]


s = [64, 34, 25, 12, 22, 11, 90]
maopao_sort(s)
print(s)

