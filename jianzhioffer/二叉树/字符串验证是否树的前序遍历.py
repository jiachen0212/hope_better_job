# coding=utf-8
# 字符串验证是否树的前序遍历
'''
//遍历一边str[]
//如果不是"#"就会多出一个叶子结点，如果是"#"就会减少一个叶子结点

输入: "9,3,4,#,#,1,#,#,2,#,6,#,#"
输出: true
'''

class Solution(object):
    def isValidSerialization(self, preorder):
        res = 1  # 叶节点的个数
        for val in preorder.split(','):
            if not res:
                return False
            if val == "#":
                res -= 1
            else:
                res += 1
        return not res

s = Solution()
print(s.isValidSerialization('9,3,4,#,#,1,#,#,2,#,6,#,#'))
