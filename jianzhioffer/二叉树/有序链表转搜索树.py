# coding=utf-8
# 有序链表转搜索树

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# pre=None其实还不是很理解...
class Solution(object):
    def sortedListToBST(self, head):
        if not head:
            return None
        fast, slow, pre = head, head, None

        # 快慢指针找到中点，作为二叉搜索树的根节点  然后左右边递归
        while fast and fast.next:
            pre = slow
            fast = fast.next.next
            slow = slow.next
        # 出循转是因为fast到尾了  此时slow正好是mid
        root = TreeNode(slow.val)
        if pre:
            pre.next = None
            root.left = self.sortedListToBST(head)  # 注意这里是head！！
            root.right = self.sortedListToBST(slow.next)
        return root