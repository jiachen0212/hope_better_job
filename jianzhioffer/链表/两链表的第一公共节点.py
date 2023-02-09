#coding=utf-8
# 寻找两个链表的第一个公共节点
# p1 p2两个指针,链长更短的那个首先走到链尾(假设是p1).则把p1指向另一个链(即更长的那个链),然后p1p2一起走直到p2也走到链尾.
# 这时候p1在长链上走的就是两个链长间的差值.再把p2指向短链首,然后p1p2一起走,直到遇到第一个相同的节点,即为所求.



# 牛客 ac
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        p1, p2 = pHead1, pHead2
        while p1 != p2:
            p1 = pHead2 if not p1 else p1.next  # pHead2 if not 表示 pHead2 是空的,即走到了链2的尾巴,则把它指到链1首, 否则就继续在p1上next
            p2 = pHead1 if not p2 else p2.next
        return p1