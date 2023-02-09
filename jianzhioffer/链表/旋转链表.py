# coding=utf-8
# 旋转链表
'''
输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL
https://leetcode-cn.com/problems/rotate-list
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 先把链表首尾相连，再找到位置断开循环

class Solution(object):
    def rotateRight(self, head, k):
        if not head or not head.next:
            return head
        start, end, llen = head, None, 0

        # 得到链表尾节点end 并统计好了链表的长度
        while head:
            end = head
            head = head.next
            llen += 1

        end.next = start  # 链表首尾相连
        pos = llen - k % llen
        while pos > 1:
            start = start.next
            pos -= 1
        res = start.next
        start.next = None  # 也就是pos之前的直接搬过来，start之后就是新链表的尾
        return res
