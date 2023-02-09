# coding=utf-8
# 二叉树的最大最小深度



###### 最大深度
# https://blog.csdn.net/qiubingcsdn/article/details/82381788
# 递归
def maxDepth(self, root):
        if not root:
            return 0
        left = self.maxDepth(root.left)+1
        right = self.maxDepth(root.right)+1

        return max(left, right)

# 非递归
# 层次遍历的感觉
def maxDepth(self, root):
        if not root:
            return 0
        curLevelNodeList = [root]
        length = 0
        while curLevelNodeList:
            tempNodeList = []
            for node in curLevelNodeList:
                if node.left:
                    tempNodeList.append(node.left)
                if node.right:
                    tempNodeList.append(node.right)
            curLevelNodeList = tempNodeList
            length += 1
        return length




########### 最小
# https://blog.csdn.net/qiubingcsdn/article/details/82419605
# 递归
def minDepth(self, root):
    if not root:
        return 0
    if not root.left and root.right is not None:
        return self.minDepth(root.right)+1
    if root.left is not None and not root.right:
        return self.minDepth(root.left)+1
    left = self.minDepth(root.left)+1
    right = self.minDepth(root.right)+1
    return min(left,right)


# 非递归
def minDepth(self, root):
    if not root:
        return 0
    curLevelNodeList = [root]
    minLen = 1
    while curLevelNodeList:
        tempNodeList = []
        for node in curLevelNodeList:
            if not node.left and not node.right:
                return minLen
            if node.left:
                tempNodeList.append(node.left)
            if node.right:
                tempNodeList.append(node.right)
        curLevelNodeList = tempNodeList
        minLen += 1
    return minLen
