#coding=utf-8

def max_len_no_repeat_substr(s):
    res = 0
    if s is None or len(s) == 0:
        return res
    d = {}  # key是s中的字符值,val是它该字符出现的下标
    start = 0
    for i in range(len(s)):
        if s[i] in d and d[s[i]] >= start:
            # print s[i], d[s[i]]
            # 这里用字典的好处是记录了之前的元素值及对应的下标,当出现重复的元素时候可以找到它上一次出现的位置,然后在这个位置的基础上右移一位
            # 而不是从到到位的遍历每一个字符..较之方法二的p1逐个移动高效很多..
            start = d[s[i]] + 1  # 出现重复字符,则在这个重复的后面移一位
        tmp = i - start + 1  # tmp存储当前的最大不重复字符串长度  start表示最大不重复串长的首字符的下标
        d[s[i]] = i  # key是s中的字符值,val是它该字符出现的下标
        res = max(res, tmp)
    return res






