# coding=utf-8
'''
反转字符串中每个单词的字符顺序
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
'''
class Solution(object):
    def reverseWords(self, s):
        return ' '.join(i[::-1] for i in s.split())
