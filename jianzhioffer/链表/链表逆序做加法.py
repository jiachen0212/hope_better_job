# coding=utf-8
# 两链表逆序做加法
# 需要一个进位位
# / 和 % 的使用
# 输出的 "和" 链表也是逆序的

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        ans = ListNode(0)  # init一个节点
        r = ans
        jw = 0  # 进位值
        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            tmp = x + y + jw
            jw = tmp / 10
            r.next = ListNode(tmp % 10) # 本位的值
            r = r.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if jw > 0:
            r.next = ListNode(1)  # 最后那位可能存在进位，9+9最大可能是18 所以r.next再给1就ok
        return ans.next