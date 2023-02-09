# coding=utf-8
# 无向图是否构成树
'''
给出n = 5 并且 edges = [[0, 1], [0, 2], [0, 3], [1, 4]], 返回 true.

给出n = 5 并且 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], 返回 false.
'''
'''
### 深度优先搜索  版本1
public class Solution {
    private boolean[] visited;
    private int visits = 0;
    private boolean isTree = true;
    private void check(int prev, int curr, List<Integer>[] graph) {
        if (!isTree) return;
        if (visited[curr]) {
            isTree = false;
            return;
        }
        visited[curr] = true;
        visits ++;
        for(int next: graph[curr]) {
            if (next == prev) continue;
            check(curr, next, graph);
            if (!isTree) return;
        }

    }
    public boolean validTree(int n, int[][] edges) {
        visited = new boolean[n];
        List<Integer>[] graph = new List[n];
        for(int i=0; i<n; i++) graph[i] = new ArrayList<>();
        for(int[] edge: edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }
        check(-1, 0, graph);
        return isTree && visits == n;
    }
}
'''



# 版本2
# 深度优先遍历可访问到的节点和边，如果是n个节点和n-1条边，那就是树
# https://blog.csdn.net/u013887008/article/details/49429485
'''
bool isTree(Graph &G){
    for(i =0; i < G.VexNum; i++)
        visited[i] = False;
    int Vnum = 0;
    int Enum = 0;
    DFS(G, 1, Vnum, Enum, visited);
    if(Vnum==G.vexnum&&Enum==2*(G.vexnum-1))
        return True;
    else
        return False;
}

void DFS(Graph &G, int v, int &Vnum, int &Enum, int visited[]){
    visited[v] = True;
    Vnum++;
    int w = FirstNeighbor(G,v);
    while(w！=-1){
        Enum++;
        if(!visited[w])
            DFS(G, w, Vnum, Enum, visited);
        w = NextNeighbor(G, v, W);
    }
}
'''


# 版本3
# https://blog.csdn.net/jmspan/article/details/51111048
class Solution(object):
    def istree(self, n, edges):
        if len(edges) != n-1:
            return False
        roots = [i for i in range(n)] # n个节点
        for i in range(n-1):
            root1 = self.root(roots, edges[i][0])
            root2 = self.root(roots, edges[i][1])
            if root1 == root2:  # 自我闭环，肯定不是树了
                return False
            roots[root2] = root1
        return True

    def root(self, roots, ind):
        if ind == roots[ind]:
            return ind
        return self.root(roots, roots[ind])


s = Solution()
ans = s.istree(5, [[0, 1], [3, 2], [0, 3], [3, 4]])
print(ans)
