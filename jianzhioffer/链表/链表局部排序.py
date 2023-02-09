# coding=utf-8
# 链表局部排序   leetcode86
'''
给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。

你应当保留两个分区中每个节点的初始相对位置。


输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5

'''
class Solution(object):
    def partition(self, head, x):
        # init两个虚拟指针 p1负责<x的移动 p2负责>=x的移动
        p1 = ListNode(-1)
        p2 = ListNode(-1)
        tmp_p1 = p1
        tmp_p2 = p2
        while head:
            if head.val < x:
                p1.next = ListNode(head.val) # 注意是p1.next
                p1 = p1.next
            else:
                p2.next = ListNode(head.val)
                p2 = p2.next
            head = head.next
        p1.next = tmp_p2.next  # 移好的p1接到p2前去
        return tmp_p1.next

