class Solution:
    def __init__(self, w: List[int]):
        self.k = [w[0]]
        self.total = w[0]
        for i in range(1, len(w)):
            self.k.append(self.k[-1] + w[i])
            self.total += w[i]
    def pickIndex(self) -> int:
        rand = random.randint(1, self.total)  #从total中随机找一个数
        return bisect.bisect_left(self.k, rand)
