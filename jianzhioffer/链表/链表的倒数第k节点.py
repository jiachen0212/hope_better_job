#coding=utf-8

# 寻找链表的倒数第k个节点.
# 使用两个指针分别从链头出发: p1先走k-1步,然后p2再出发,所以p1p2之间相差k-1步.p1接着走到链尾的时候,p2指着的就是倒数第k个节点了.
# 因为p12间一直是相差k-1步的... find_last_k()函数
# 同样的使用两个指针还可以做到遍历一次链表就指向链的中间节点. 即两指针一起从链头出发,一个一次走两步,一个一次走一步.当走的快的到达链尾,走的慢的就是中间节点了
# find_mid()函数  并且在链表的奇偶长度上要做一点小区分.
# 两个指针的灵活使用真的很精髓!!!


# 前后指针  p1先走k步，然后p2走  p1到链尾了，p2就是倒数第k节点了

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        if head == None or k <= 0:
            return None
        # init p1 p2
        p1, p2 = head, head    # 设定两个指针，一个先走k-1步另一个再开始走，当先走的到尾了，后走的正好是倒数第k节点
        # p1先走k步
        while k  > 1:
            if p1.next != None:
                p1 = p1.next
                k -= 1
            else:
                return None    # 链长不够k
        while p1.next != None:
            p1 = p1.next
            p2 = p2.next   # 好了现在p2开始走
        return p2

