# coding=utf-8
# 堆排序
# https://www.jianshu.com/p/0d383d294a80
# 时复 nlogn

# 堆维护的时间复杂度 klogk
# 大根堆   父>子
# 目的就是把父调大，让最顶元素最大
def heap_adjust(array, start, end):
    # start到end构成一棵父>子的树 start end 是当前的最顶和最末
    # 即需要换顺序的俩数
    temp = array[start]   # 堆维护前的堆顶  max
    child = 2 * start
    while child <= end:
        if child < end and array[child] < array[child + 1]:
            child += 1
        if temp >= array[child]:
            break
        array[start] = array[child]  # 把父调成子中更大的
        start = child  # 好, 更新下一层 所以start变成child
        child *= 2  # 好 下一层
    array[start] = temp  # 堆维护前start的值 和 child中最大值位置的值互换
    # line21 和 line18


def heap_sort(array):
    # 首次建堆
    first = len(array) // 2 - 1
    for start in range(first, -1, -1):
        heap_adjust(array, start, len(array) - 1)

    # 将最大的数放到堆的最后一个位置，并继续调整排序
    # 注意end的值 从最大依次渐小  因为后面调整好的元素就不用管了
    # 这点和冒泡一样
    for end in range(len(array)-1, 0, -1):
        array[0], array[end] = array[end], array[0]
        heap_adjust(array, 0, end-1)

nums = [3,6,5,-1,7,8,1]
heap_sort(nums)
print(nums)
