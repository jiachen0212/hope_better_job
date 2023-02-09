#coding=utf-8
# 求二叉树的深度. 递归的求左右子树的深度,然后取它们的较大值再加1就得到树深

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 牛客 ac  递归方法  深度
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot == None:
            return 0
        lDepth = Solution.TreeDepth(self, pRoot.left)
        rDepth = Solution.TreeDepth(self, pRoot.right)
        return max(lDepth, rDepth) + 1



# 宽度
import queue
class Node:
    def __init__(self,value=None,left=None,right=None):
        self.value=value
        self.left=left
        self.right=right


def treeWidth(tree):
    curwidth=1
    maxwidth=0
    q=queue.Queue()
    q.put(tree)
    while not q.empty():
        n=curwidth
        for i in range(n):
            tmp=q.get()
            curwidth-=1
            if tmp.left:
                q.put(tmp.left)
                curwidth+=1
            if tmp.right:
                q.put(tmp.right)
                curwidth+=1
        if curwidth>maxwidth:
            maxwidth=curwidth
    return maxwidth

if __name__=='__main__':
    root=Node('D',Node('B',Node('A'),Node('C')),Node('E',right=Node('G',Node('F'))))
    width=treeWidth(root)
    print('width:',width)









# 非递归计算树的深度
int depth_Non_recursive(bitree bt)//利用队列进行非递归算法
{
    queue<bitree> q;
    bitree p=bt;
    int level=0,len;
    if(!bt)
        return 0;
    q.push(p);
    while(!q.empty())//每次只有把在同一层的所有结点出队以后才level++，因此要知道当前队列的长度，用len表示
    {
        level++;
        len=q.size();//当前队列长度就代表这一层的结点个数
        while(len--)
        {
            p=q.front();
            q.pop();
            if(p->lchild)
                q.push(p->lchild);
            if(p->rchild)
                q.push(p->rchild);
        }
    }
    return level;
}

# https://blog.csdn.net/shijie97/article/details/79775399