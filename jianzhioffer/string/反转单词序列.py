#coding=utf-8
# 翻转单词序列  栗子: i am a student. >> student. a am i
# 方法一: 翻两次:首先把所有的字符均翻转,然后再把每个单词翻回来...
# python里的reserve()函数很好用... student >> tneduts
# 辅助使用python的切片...把一个个反单词单词割出来...

'''
leetcode 151
输入: "the sky is blue"
输出: "blue is sky the"
'''
def fanzhuankeys(s):
    s = list(s)  # 先把str变成list,方便使用reverse()  reverse()函数没有返回值,直接是逆序s的作用..
    s.reverse()
    s = ''.join(s)
    news = ''
    for key in s.split(' '):  # key就是每一个的单词了..
        key = list(key)
        key.reverse()
        key = ''.join(key)
        news += key
        news += ' '
    return news

s = 'hello world!'
res = fanzhuankeys(s)
print res   # world! hello



#######  牛客 bt ac 版
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # s.split(" ") 直接割出每一个单词，然后[::-1]单词级别的反转
        return " ".join(s.split(" ")[::-1])

s = Solution()
res = s.ReverseSentence('we; tonight! you;')


###### leetcode 上类似的题：
class Solution(object):
    def reverseWords(self, s):
        s = s.strip(' ')
        words = s.split(' ')
        res = ''
        for word in words[::-1]:
            if word:
                res += word
                res += ' '
        res = res.rstrip(' ')
        return res




##### 牛客上的另一道题：
# 翻转中间由各种符号隔开的字符串
def rever(s):
    tmp = ''
    res = ''
    for ch in s:
        if ch.isalpha():
            tmp += ch
        else:
            res += (tmp[::-1])
            res += ch  # 这个东东虽然不是字母，但也把它加进来
            tmp = ''
    return res

ans = rever('hello word!')
print(ans)  # ew; thginot! uoy;


