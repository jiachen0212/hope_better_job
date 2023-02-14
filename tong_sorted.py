# coding:utf-8
def bucketSort(nums):
  max_num = max(nums)
  bucket = [0]*(max_num+1)
  for i in nums:
    bucket[i] += 1
  sort_nums = []
  for j in range(len(bucket)):
    if bucket[j] != 0:
        sort_nums.append(j)
  return sort_nums

nums = [5,6,3,2,1,65,2,0,8,0]
print(bucketSort(nums))