# coding=utf-8
#### 牛客神代码  ac版


# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        ans = n
        # 这句and是精髓，任意a and b， 返回的都是b......
        temp = ans and self.Sum_Solution(n - 1)
        ans = ans + temp
        return ans



s = Solution()
res = s.Sum_Solution(11)
print(res)