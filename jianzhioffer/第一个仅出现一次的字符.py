# -*- coding:utf-8 -*-
# 用哈希表做，就不存在dict的那个遍历无序的苦恼...
class Solution:
    def FirstNotRepeatingChar(self, s):
        if len(s) < 1:
            return -1
        ls=[0]*256    # 创建一个长为256的哈希表
        for i in s:     #遍历字符串,下标为ASCII值,值为次数
            ls[ord(i)] += 1    # ord()函数为取字符的ascll格式，也即是哈希的key   value为出现的次数

        for j in s:      # 再遍历下列表,找到出现次数为1的字符并输出位置
            if ls[ord(j)]==1:
                return s.index(j)
                break



# 牛客ac  但是用字典不是最佳方案
# -*- coding:utf-8 -*-
class Solution:
    # 用字典不是很好，因为字典的遍历是无序的...
    def FirstNotRepeatingChar(self, s):
        if s == "":
            return -1
        if len(s) == 1:
            return 0
        dic = {}
        for i in range(len(s)):
            if s[i] not in dic.keys():
                dic[s[i]] = 1
            else:  # 表明dic中已经有s[i]个key
                dic[s[i]] += 1
        cache = []
        for key, value in dic.items():  # 同时遍历key和value.  items()
            if dic[key] == 1:
                cache.append(s.index(key))
        return min(cache)


# 2021 update
class Solution(object):
    def firstUniqChar(self, s):
        if len(s) < 1:
            return -1
        # 预设一个256长度的list,使用哈希表把char转为[0,255]范围内的int, (ord()函数实现)
        # 则每个index代表一个char, index上的value则是出现次数
        # 无需dict,节约内存啊~
        hx = [0]*256
        for char in s:
            hx[ord(char)] += 1
        # hx list 创建完毕
        for i, char in enumerate(s):
            if hx[ord(char)] == 1:
                return i
        return -1  # s可能没有只出现一次的char, 故-1兜个底



