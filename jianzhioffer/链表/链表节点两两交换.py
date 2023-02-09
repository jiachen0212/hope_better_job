# coding=utf-8

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        if not head or not head.next:
            return head
        p = head
        fast = p
        while p and p.next:
            fast = p.next.next
            # 换值
            tmp = p.val
            p.val = p.next.val
            p.next.val = tmp
            # 换节点
            p = fast
        return head