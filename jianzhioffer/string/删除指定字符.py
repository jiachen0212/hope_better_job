# coding=utf-8
# https://www.cnblogs.com/liushaobo/p/4373752.html

# 删除指定字符
# 利用类似构建哈希表的方法

def deleteChars(s1, s2):
    if not s2:
        return s1
    if not s1:
        return ''
    hash_map = [0]*256

    for ch in s2:
        hash_map[ord(ch)] = 1   # 把s2中字符的ascall值作为index，填充hasmap

    res = ''
    for ch1 in s1:
        if not hash_map[ord(ch1)]:
            res += ch1
    return res

print deleteChars('They are students.', 'aeiou')

