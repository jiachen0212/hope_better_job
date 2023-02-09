# coding=utf-8
# leetcode： https://leetcode-cn.com/problems/score-of-parentheses/

# 用stack做

def scoreOfParentheses(S):
    '''
    思路：tmp计分，栈设计(括号里的分数)，最终结算分数为：2*括号里的分数(当括号里的分数为0时则分数为1)\
    测试用例：
    前提：字符只有()，并且最终可以消除
    '''
    stack = [0]
    for char in S:
        if (char == "("):
            stack.append(0)
        else:
            score = stack.pop()
            stack[-1] += (1 if (score == 0) else 2 * score)  # if (score == 0) 代表char是'(',不构成括号的另一半
    return stack[0]


s = '()()'
ans = scoreOfParentheses(s)
print(ans)