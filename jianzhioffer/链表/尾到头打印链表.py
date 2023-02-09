# 牛客ac
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        res = []
        stack = []   # 存储栈
        while listNode:
            stack.append(listNode.val)   # 把节点的值放进栈
            listNode = listNode.next  # 移动到下一个节点
        if stack:   # 堆栈不为空, 即上面传进来的链表是不空的.
            while stack:
                res.append(stack.pop(-1))    # 开始把堆栈中的东西推出去打印出来了
        else:
            res = []
        return res



# 递归 c++版  牛客ac
class Solution{
    public:
    vector<int> value;
    vector<int> printListFromTailToHead(ListNode* head){
        ListNode *p = NULL;
        p = head;     // 头节点开始遍历
        if (p != NULL){
            if (p->next != NULL){
                printListFromTailToHead(p->next);
            }
            value.push_back(p->val);   //值放入栈
        }
        return value;
    }
};

