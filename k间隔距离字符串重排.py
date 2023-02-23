class Solution:
    '''
    重排字符串 让相同字符间隔至少为k
    '''
    def rearrangeString(self, s: str, k: int) -> str:
        from collections import Counter
        import heapq
        if k <= 1: return s
        c = Counter(s)
        n = len(s)
        heap = [(-v, k) for k, v in c.items()]
        heapq.heapify(heap)
        res = ""
        while heap:
            tmp = []
            for _ in range(k):
                if not heap:return res if len(res) == n else ""
                num, alp = heapq.heappop(heap)
                num += 1
                res += alp
                if num != 0:
                    tmp.append((num, alp))
            for t in tmp:
                heapq.heappush(heap, t)
        return res
