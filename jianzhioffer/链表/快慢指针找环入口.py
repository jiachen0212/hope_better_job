# coding=utf-8

####### 牛客 ac
# 快慢指针找环入口

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
'''
首先快慢指针，fast=2*slow  fast和slow相遇，则fast比slow多走了一个环才会相遇的
所以slow此时是走到了一个环长度的位置
那现在把fast放到head处，slow fast同步走，下一次相遇是slow比fast多走一个环长度的位置
即正好是环的入口啊...
'''
class Solution:
    def detectCycle(self, pHead):
        if not pHead or not pHead.next or not pHead.next.next:
            return None

        fast = pHead.next.next
        slow = pHead.next
        # 判断有无环
        while fast!=slow:
            if fast.next and fast.next.next:
                fast=fast.next.next
                slow=slow.next
            else:
                return -1   # 无环

        # 跳出循环说明有环
        fast = pHead # slow在fast前环长度
        while fast!=slow:
            fast=fast.next
            slow=slow.next
        return slow   # 在环入口相遇



# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# head = ListNode(1)
# p1 = ListNode(2)
# p2 = ListNode(3)
# p3 = ListNode(4)
# p4 = ListNode(5)
# head.next = p1
# p1.next = p2
# p2.next = p3
# p3.next = p4
# p4.next = p2

# s = Solution()
# res = s.detectCycle(head)
# print res.val