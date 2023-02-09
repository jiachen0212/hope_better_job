# -*- coding:utf-8 -*-
# 牛客 ac 版本
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if not s and not pattern:
            return True
        if not pattern:
            return False
        if len(pattern)==1:
            if s and (s[0]==pattern[0] or pattern[0]=='.'):#()()()()
                return self.match(s[1:],pattern[1:])
            else:
                return False
        else:
            if pattern[1] != '*':  # pattern 第二个字符不等于"*"的情况
                if s[0]==pattern[0] or pattern[0] == '.':
                    return self.match(s[1:],pattern[1:])
                else:
                    return False
            else:    # pattern 第二个字符等于"*"的情况
                if pattern[0] == '.':
                    match = 0
                    for i in range(len(s)+1):
                        match = self.match(s[i:],pattern[2:])
                        if match:
                            return True
                    return False
                else :
                    match = 0
                    for i in range(len(s) + 1):
                        match = self.match(s[i:],pattern[2:])  # pattern[2:]：pattern后移2个字符
                        if match:
                            return True
                        if i < len(s) and s[i]!= pattern[0]:  # pattern[2:]：pattern保持不动
                            break
                    return False
