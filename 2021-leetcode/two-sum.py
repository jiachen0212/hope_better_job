target = 6
nums = [3, 2, 6, 0, 9]
def TwoSum(nums, target):
    listlen = len(nums)
    for i in range(0, listlen - 1):
        for j in range(i + 1, listlen):
            if nums[i] + nums[j] == target:
                return (i, j)

a, b = TwoSum(nums, target)
print (a, b)

