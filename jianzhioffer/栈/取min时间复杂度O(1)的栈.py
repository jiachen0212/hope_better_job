# 牛客 ac
# -*- coding:utf-8 -*-
# 用一个辅助栈和一个最小值寄存容器帮忙
# 最小值寄存容器的目的是决定，辅助栈内压入的是新进来的数还是之前保存下来的最小值
# 推出的话，推的是辅助栈的值  实现的栈的min函数，的时间复杂度为：O(1)

class Solution:
    def __init__(self):
        self.stack = []
        self.assist = []

    def push(self, node):
        min = self.min()
        if not min or node < min:
            self.assist.append(node)
        else:
            self.assist.append(min)
        self.stack.append(node)

    def pop(self):
        if self.stack:
            self.assist.pop()
            return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]

    def min(self):
        if self.assist:
            return self.assist[-1]