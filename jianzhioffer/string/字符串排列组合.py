# coding=utf-8
########## 牛客 ac
# 对字符串中的字符进行排列
# 要擅用递归...
class Solution:
    def __init__(self):
        self.result = []
    def Permutation(self, ss):
        s = []
        to = len(ss)
        # ss='abc'  s=['a','b','c']
        for i in range(to):
            s.append(ss[i])
        self.PermutationHelper(s, 0, len(ss))
        self.result = list(set(self.result))
        self.result.sort()  # sort()list的排序函数
        return self.result

    def PermutationHelper(self, ss, fro, to):
        if to <= 0:
            return
        if fro == to-1:  # 执行完这一句后,会跳出self.PermutationHelper(ss, fro + 1, to).然后执行@@@代码.
            self.result.append(''.join(ss))  # 这个ss是换好位置的string了
            # print '$$$', self.result
        else:
            for i in range(fro, to):
                self.swap(ss, i, fro)  # 这个函数是为了改变ss
                # print(ss, '####', 'i:', i, 'fro:', fro)  # &&&
                self.PermutationHelper(ss, fro + 1, to)  # 当调用这个函数并且fro+1=len(s)-1,则函数添加完新str进入res后,
                # 会立即退出来执行它下面的self.swap(ss, fro, i),执行self.swap(ss, fro, i)时候fro和i仍相等.但i会继续+1使得fro和i之间相差1.
                # 这个时候程序返回到第一句 &&& self.swap(ss, i, fro)这里.实现s的元素位置变化. 且这个时候的fro也是fro+1=len(s)-1的.所以这个交换后的str可被立即加入res.
                # 直到内层的i被遍历完(至len(s)-1). 然后外层的i+1继续...
                self.swap(ss, fro, i)  # @@@
                # print ss, '%%%%', 'i:', i, 'fro:', fro
                # 执行这一句,fro=len(s)-2即刚添加完一个新str进入res.或者是最外层的i在进行循环.
    def swap(self, str, i, j):
        str[i], str[j] = str[j], str[i]
s = Solution()
print s.Permutation('abcd')

# 另一种更好的理解方法是：
# 不这样理解的话,看程序的流程走真的特别糊不知道它的代码思想...
'''
if fro == to - 1:
    self.result.append(''.join(ss))
else:
    for i in range(fro, to):  # fro起始是0,固定的是第一个位置元素,
        self.swap(ss, i, fro) # &&& 把fro第一个元素依次和后面的值交换位置.包括和它自己. 所以执行 self.swap(ss, i, fro)
        self.PermutationHelper(ss, fro + 1, to)  # 完了之后呢,把fro这个指针后移一位变成fro,并且递归调用函数PermutationHelper
        # 意思就是也和之前的思想一样,只是我拿去依次交换的元素后移了一位而已.
        self.swap(ss, fro, i) # 这一句是因为上面使用递归函数的时候fro+1是参数,则函数一进来立马执行了if fro == to - 1:  self.result.append(''.join(ss))
        也就是说没有执行 self.swap(ss, i, fro)&&&, 即fro=len(s)-2位置处的值没有进行和后面元素的交换..所以这里需要补充这一句!!!
'''


