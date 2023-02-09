# coding=utf-8
'''
leetcode92 ac
反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。

输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL

'''

class Solution(object):
    # reverse()辅助函数
    def reverse(self, head):
        pre = None
        while head:
            nxt = head.next
            head.next = pre  # 反指针操作
            pre = head
            head = nxt
        return pre

    # 找第k个节点 因为链表不连续 所以只能一直next的找 不像数组可以直接index得到
    def findkth(self, head, k):
        for i in range(k):
            if head:
                head = head.next
            else:
                return None
        return head

    def reverseBetween(self, head, m, n):
        dummy = ListNode(-1)
        dummy.next = head  # dummy是新init的一个head之前的节点
        mth_pre = self.findkth(dummy, m-1)
        mth = mth_pre.next
        nth = self.findkth(dummy, n)
        nth_next = nth.next
        nth.next = None

        self.reverse(mth)
        # 下面两句是精髓 画个链表图会清晰很多
        mth_pre.next = nth
        mth.next = nth_next
        return dummy.next




