# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        # char_list 是个哈希表，key是char，value是进来的字符的index，理解为序号吧..
        self.char_list = [-1 for i in range(256)]    # 预设是这个字符串流长256
        self.index = 0  # 记录当前字符的个数，可以理解为输入的字符串中的下标

    # ord(char)是得到char对应的ASCII码；chr(idx)是得到ASCII位idx的字符

    def FirstAppearingOnce(self):
        min_value = 500
        min_idx = -1
        for i in range(256):
            if self.char_list[i] > -1:   # > -1 表示读入了字符且暂时还没被修改成-2 即还是第一次呢...
                if self.char_list[i] < min_value:
                    min_value = self.char_list[i]  # 应该是为了：在数组中找到 >0 的最小值 吧...

                    min_idx = i
        if min_idx > -1:
            return chr(min_idx)   # 返回这个第一次出现的char
        else:
            return '#'

    def Insert(self, char):
        if self.char_list[ord(char)] == -1:   # -1 还是初始化时候的值，即这个字符第一次出现
            self.char_list[ord(char)] = self.index   # 第一次出现，将对应元素的值改为下标(这个下标就是字符的index 类似序号这样理解吧)

        elif self.char_list[ord(char)] == -2: # 如果已经出现过两次了，则不修改
            pass

        else:   # 如果仅出现过一次，则进行修改，修改为-2
            self.char_list[ord(char)] = -2
        self.index += 1   # index++ ok 下一个字符快来...