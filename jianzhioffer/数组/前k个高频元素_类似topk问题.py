# coding=utf-8
# 前k个高频元素

class Solution:
    def topKFrequent(self, nums, k):
        from collections import Counter
        import heapq
        count = Counter(nums)  # Counter({1: 3, 2: 2, 3: 1})
        num_set = []
        heap = []
        for i in count:
            num_set.append(i)

        for i in num_set[:k]:
            heapq.heappush(heap, (count[i], i))  # heap:[(2, 2), (3, 1)]

        for i in num_set[k:]:
            if count[i] > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (count[i], i))
        ans = []
        for pair in heap:  # 这个pair是数组元素和它出现的次数
            ans.append(pair[1])

        return ans
s = Solution()
res = s.topKFrequent([1,1,1,2,2,3],2)


