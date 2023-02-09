#coding=utf-8
################# 牛客 ac
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        # 开始遍历链表，依次的反转
        pre = None
        cur = pHead
        while cur:
            tmp = cur.next
            cur.next = pre   # 只需要进行一次next指针指向
            # 下面两行是两节点的后移操作
            pre = cur   # pre指针后移一位
            cur = tmp   # cur也后移一位
        return pre


####  me test
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
head = ListNode(1)  # 测试代码
p1 = ListNode(2)  # 建立链表1->2->3->4->None
p2 = ListNode(3)
p3 = ListNode(4)
head.next = p1
p1.next = p2
p2.next = p3

s = Solution()
res = s.ReverseList(head)
while res:
    print res.val
    res = res.next



