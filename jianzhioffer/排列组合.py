# coding=utf-8

# string的全排列
# 递归完成
def perm(s=''):
    if len(s)<=1:
        return [s]
    res=[]
    for i in range(len(s)):
        for j in perm(s[0:i]+s[i+1:]):
            res.append(s[i]+j)
    return res
print perm('123')

# 非递归完成
def Swap(n,a,b):
    n[a],n[b] = n[b],n[a]
    return None
def Reverse(n,begin):
    if len(n) > begin:
        i = begin
        j = len(n)-1
        while i < j:
            Swap(n,i,j)
            i += 1
            j -= 1
    return n

def FindMin(n,i):
    j = len(n)-1
    k = i + 1
    while j > i:
        if n[j] > n[i] and n[j] < n[k]:
            k = j
        j -= 1
    return k

def Permut(n):
    count = 0
    j = len(n) -1
    if j < 1:
        return n
    else :
        count += 1
        while j >= 1:
            i = j - 1
            if n[i] < n [j] :
                k = FindMin(n,i)
                Swap (n,i,k)
                Reverse (n,j)
                j = len(n) - 1
                count += 1
            else :
                j -= 1
    print count



# list的全排列
# 回溯法
class Solution:
    def permute(self, nums):
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
        backtrack(nums, [])
        return res