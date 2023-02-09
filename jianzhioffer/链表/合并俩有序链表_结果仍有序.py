#coding=utf-8
# 合并两个排序链表,使结果依然有序.


# 非递归版  牛客ac
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def mergeTwoLists(self, pHead1, pHead2):
        tmp = ListNode(0)
        phead = tmp
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                tmp.next = pHead1
                pHead1 = pHead1.next
            else:
                tmp.next = pHead2
                pHead2 = pHead2.next
            tmp = tmp.next
        if pHead1:
            tmp.next = pHead1
        if pHead2:
            tmp.next = pHead2
        return phead.next









