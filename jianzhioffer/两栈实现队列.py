# 牛客ac
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []  # 入数据
        self.stack2 = []  # 出数据，靠stack1弹出的数据弹入stack2

    def push(self, node):
        self.stack1.append(node)

    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        elif not self.stack1:    # stack1空，那就直接none了，没数据可删
            return None
        else:   # stack1有数据，stack2为空
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()

