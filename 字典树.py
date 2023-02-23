
# 单词搜索  矩阵中分布字符, 上下左右邻域, 求可以拼接出的单词
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        visited = [[0]*len(board[0]) for _ in range(len(board))]
        starts = set()
        max_len = 0
        maps = set()
        for word in words:
            max_len = max(len(word),max_len)
            starts.add(word[0])
            for i in range(len(word)):
                maps.add(word[:i+1])
        words = set(words)
        ds = [[-1,0],[1,0],[0,1],[0,-1]]
        res = []
        def dfs(i,j,cur_word):
            if cur_word in words: res.append(cur_word)#输出结果
            if len(cur_word)>=max_len: return#剪枝
            #深度优先搜索
            for d in ds:
                new_x,new_y = i+d[0],j+d[1]
                if 0<=new_x<len(board) and 0<=new_y<len(board[0]):
                    if visited[new_x][new_y] == 0:
                        new_word = cur_word+board[new_x][new_y]
                        if new_word in maps:#剪枝
                            visited[new_x][new_y]=1
                            dfs(new_x,new_y,new_word)
                            visited[new_x][new_y]=0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in starts:#剪枝
                    visited[i][j]=1
                    dfs(i,j,board[i][j])
                    visited[i][j]=0
        return list(set(res))#去重



