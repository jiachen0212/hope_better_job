# coding=utf-8
# 合法括号
'''
()[]{} true
([)] false
'''
class Solution:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''



# Q2
# https://leetcode-cn.com/problems/valid-parenthesis-string/submissions/

'''
任何左括号 ( 必须有相应的右括号 )。
任何右括号 ) 必须有相应的左括号 ( 。
左括号 ( 必须在对应的右括号之前 )。
* 可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
一个空字符串也被视为有效字符串。
'''
# 需要注意一个 ) 必须前面有和它对应的(
# 所以借用堆栈实现  看到右括号 就去左括号或*栈里减一个
# 最后还要判断(是否在*后面出现，是的话就False   所以栈存的是index
class Solution(object):
    def checkValidString(self, s):
        if not s:
            return True
        l = len(s)
        le = []
        star = []
        for i in range(l):
            if s[i] == '(':
                le.append(i)  # 保存index 后面需要用到的
            elif s[i] == '*':
                star.append(i)
            else:
                if not le and not star:
                    return False
                elif le:
                    le.pop()
                else:
                    star.pop()
        while le and star:
            if le[-1] > star[-1]:
                return False
            le.pop()
            star.pop()
        return not le

