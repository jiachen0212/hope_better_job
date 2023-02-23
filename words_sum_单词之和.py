class MapSum:
    '''
    MapSum() 初始化 MapSum 对象
    void insert(String key, int val) 插入 key-val 键值对，字符串表示键 key ，整数表示值 val 。如果键 key 已经存在，那么原来的键值对将被替代成新的键值对。
    int sum(string prefix) 返回所有以该前缀 prefix 开头的键 key 的值的总和。
    '''
    def __init__(self):
        self.tree = {}

    def insert(self, key: str, val: int) -> None:
        cur = self.tree
        i = 0
        for i in range(len(key)):
            if key[i] in cur:
                cur = cur[key[i]]
            else:
                cur[key[i]] = {}
                cur = cur[key[i]]
        cur['val'] = val   #表示此处的值

    def sum(self, prefix: str) -> int:
        cur = self.tree
        for n in prefix:
            if n not in cur: return 0  #不存在这个前缀
            cur = cur[n]
        Q = deque([cur])
        ret = 0
        while Q:
            curr = Q.popleft()
            for key in curr:
                if key == 'val':
                    ret += curr[key]
                else:
                    Q.append(curr[key])
        return ret
