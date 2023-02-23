class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # 多余的边 并查集 
        n = len(edges)
        parent = list(range(n + 1))

        def find(index: int) -> int:
            if parent[index] != index:
                parent[index] = find(parent[index])
            return parent[index]
        
        def union(index1: int, index2: int):
            parent[find(index1)] = find(index2)

        for node1, node2 in edges:
            if find(node1) != find(node2):
                union(node1, node2)
            else:
                return [node1, node2]
        
        return []
