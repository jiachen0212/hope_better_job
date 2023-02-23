class Solution:
    class UnionFind():
        def __init__(self):
            # parent[0]=1, 表示0的父节点是1
            # 根节点的父节点是自己
            self.parent = list(range(26))
        
        # 用于寻找x的根节点
        def find(self, x):
            if x == self.parent[x]:
                return x
            
            # 继续向上找父节点
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        # 两个节点的合并
        def union(self, x, y):
            # x的根节点直接指向y的根节点
            self.parent[self.find(x)] = self.find(y)

    def equationsPossible(self, equations: List[str]) -> bool:
        uf = Solution.UnionFind()

        for item in equations:
            if item[1] == '=':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 相等的进行合并操作
                uf.union(x, y)

        for item in equations:
            if item[1] == '!':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 判断两节点的根节点是否相同
                if uf.find(x) == uf.find(y):
                    return False
        
        return True
