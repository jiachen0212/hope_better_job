# coding=utf-8
# leecode ac

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        new_head = ListNode(-1)  # 在头节点前再加一个节点
        new_head.next = pHead
        pre = new_head
        p = pHead   # pre->p->nex
        nex = None
        while p and p.next:
            nex = p.next
            if p.val == nex.val:
                while nex and nex.val == p.val:
                    nex = nex.next
                pre.next = nex   # pre的next直接连接nex  中间重复的都删掉了
                p = nex
            else:
                pre = p
                p = p.next
        return new_head.next

# easy 版本
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def deleteDuplicates(self, head):
        if not head:
            return None
        node = head
        while head.next:
            if head.val==head.next.val:
                head.next=head.next.next
            else:
                head=head.next
        return node