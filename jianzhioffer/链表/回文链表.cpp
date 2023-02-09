//  链表是否回文结构
// 牛客 ac

/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};*/
// 思路 把链表的前一半压进栈，然后逐个和后面的元素对比

class PalindromeList {
public:
    bool chkPalindrome(ListNode* A)
    {
        // write code here
        if(A == NULL || A->next==NULL) return true;
        int len=Length_of_List(A);  // 计算下链表的长度
        stack<int> st;   // 存放链表的前一半元素
        ListNode* p=A;
        for(int i=1;i<=len/2;i++)
        {
            st.push(p->val);  // 压栈
            p=p->next;
        }
        if(len%2==1) p=p->next;
        while(p!=NULL)
        {   // 开始比较链表后面的值是否和前面一致
            if(st.top()!=p->val)
                return false;
            st.pop();
            p=p->next;
        }
        return true;
    }
    // 这个是统计链表长度的函数
    int Length_of_List(ListNode* A)
    {
        if(A==NULL) return 0;
        ListNode* p=A;
        int count=0;
        while(p!=NULL)
        {
            count++;
            p=p->next;
        }
        return count;
    }
};
