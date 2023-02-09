# coding=utf-8
# 奇偶index重排

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        if not head or not head.next:
            return head
        odd = head
        even = head.next
        t = even   # 注意这句
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = t  # emmm...
        return head
