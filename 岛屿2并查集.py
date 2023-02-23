class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1]*n
        self.part = n

    def find(self, x):
        if x != self.root[x]:
            # 在查询的时候合并到顺带直接根节点
            root_x = self.find(self.root[x])
            self.root[x] = root_x
            return root_x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        uf = UnionFind(m*n)

        ans = []
        cnt = 0
        island = set()
        for r, c in positions:
            if (r, c) in island:
                ans.append(cnt)
                continue
            nodes = []
            for x, y in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if 0<=x<m and 0<=y<n and (x, y) in island:
                    nodes.append((x, y))
            
            if not nodes:
                cnt += 1
            else:
                roots = set([uf.find(x*n+y) for x, y in nodes])
                cnt = cnt-len(roots) + 1
                for x, y in nodes:
                    uf.union(x*n+y, r*n+c)
            ans.append(cnt)
            island.add((r, c))

        return ans
