# -*- coding:utf-8 -*-
# 剑指P169
class Solution:
    def IsPopOrder(self, pushV, popV):
        # stack中存入pushV中取出的数据
        stack=[]   # 辅助栈 作为满足推出需要压进来的数？
        while popV:
            # 如果第一个元素相等，直接都弹出，根本不用压入stack
            # 意思是压出压出压出直接的进行，不涉及压压出...之类的，就完全不需要stack这个打辅助了...
            if pushV and popV[0] == pushV[0]:
                popV.pop(0)
                pushV.pop(0)

            elif stack and stack[-1]==popV[0]:  #如果stack的最后一个元素与popV中第一个元素相等，将两个元素都弹出
                stack.pop()   # 一直压数进来就是为了满足被弹出的需求，pop()默认弹最后一个值
                popV.pop(0)   # 弹栈顶
            elif pushV:   # 如果pushV中有数据，压入stack，为后续的出栈序列做数据准备
                stack.append(pushV.pop(0))
            else:   # 上面情况都不满足，直接返回false
                return False
        return True