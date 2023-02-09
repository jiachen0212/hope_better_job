# coding=utf-8
# 重排链表
'''
给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
'''

# 用双端队列做  pop(0) and pop()
class Solution(object):
    def reorderList(self, head):
        if not head or not head.next:
            return head
        cur = head
        stack = []
        while cur:
            stack.append(cur)
            cur = cur.next
        # 跳出while了  即所有的节点都放进stack了

        cur = stack.pop(0)  # 弹出第一个
        while stack:
            cur.next = stack.pop() # next的赋值
            cur = cur.next
            if stack:
                cur.next = stack.pop(0)
                cur=cur.next
        cur.next = None   # 走到最末了