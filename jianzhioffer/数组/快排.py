# coding=utf-8
# 快排

def quick_sort(nums, left, right):
    if left >= right:
        return nums
    low = left
    high = right
    key = nums[left]
    while left < right:
        while left < right and nums[right] >= key:
            right -= 1
        # 跳出上面的while，说明右边出现小于key的值，把它放到左边去
        nums[left] = nums[right]
        while left < right and nums[left] <= key:
            left += 1
        # 跳出上面的while，说明左边出现大于key的值，把它放到右边去
        nums[right] = nums[left]
    # 跳出循环，即left>=right
    nums[left] = key
    # 现在完成了小于等于key的在左，大于key的在右
    # 那就递归的把左右分别排序好吧
    # left处等于key值
    quick_sort(nums, low, left-1)
    quick_sort(nums, left+1, high)
    return nums

nums = [5,3,3,7,1,8,1,4]
print quick_sort(nums, 0, 7)