# coding=utf-8
# 和等于target，每个组合中元素不重复
def combinationSum2(nums, target):
    nums.sort()
    table = [None] + [set() for i in range(target)]
    # [None, set([]), set([]), set([]), set([]), set([]), set([])]   target个set()
    for i in nums:
        if i > target:
            break
        for j in range(target - i, 0, -1):
            # |位运算: 只要相应位上存在1，那么该位就取1，均不为1，即为0
            table[i + j] |= {elt + (i,) for elt in table[j]}
        table[i].add((i,))
    return map(list, table[target])

res = combinationSum2([12,14,11,15], 26)
print res   # [[11, 15], [12, 14]]



