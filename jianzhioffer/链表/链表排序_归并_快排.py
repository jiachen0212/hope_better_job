# coding=utf-8
#### leetcode

# https://www.cnblogs.com/chengxiao/p/6194356.html

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 归并排序
class Solution:
    def sortList(self, head):
        if not head or not head.next:
            return head
        mid = self.get_mid(head)
        l = head
        r = mid.next
        mid.next = None
        return self.merge(self.sortList(l), self.sortList(r))

    def merge(self, p, q):
            tmp = ListNode(0)
            h = tmp
            # p q 分别是两段的头节点
            while p and q:
                if p.val < q.val:
                    h.next = p
                    p = p.next
                else:
                    h.next = q
                    q = q.next
                h = h.next
            if p:
                h.next = p
            if q:
                h.next = q
            return tmp.next

    def get_mid(self, node):
        if not node:
            return node
        fast = slow = node
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow



# 快排
# class nodelist:
#     def __init__(self,x):
#         self.val = x
#         self.next = None
class Solution:
    def sortList(self,phead):
        if not phead:
            return None
        else:
            self.quicksort(phead,None) # head and end
        return phead

    # p1 p2 节点交换
    def swap(self,node1,node2):
        tem = node1.val
        node1.val = node2.val
        node2.val = tem

    def quicksort(self,head,end):
        if head != end:
            key = head.val
            p = head
            q = head.next   # p q 两指针
            while q != end:  # q 遍历除参考值外的所有节点
                if q.val < key:  # 出现节点的值小于参考值
                    p = p.next # 先把p前移一位，再给这个位置赋予刚刚q的值
                    self.swap(p,q)# 将q的值给p  使得p遍历的节点都小于key
                q = q.next
            self.swap(head,p)  # 这一步别漏了，把key_ind和之前的head互换 然后分两段使两段均有序
            self.quicksort(head,p)
            self.quicksort(p.next,end)
