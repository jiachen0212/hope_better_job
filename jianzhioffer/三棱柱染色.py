# coding=utf-8
# 三棱柱染色问题
# 6个顶点，4个颜色，同一条边颜色不能一样.
# 264种可以染的可能

# 无重复字符的所有排列组合
def perm(s=''):
    if len(s)<=1:
        return [s]
    res=[]
    for i in range(len(s)):
        for j in perm(s[0:i]+s[i+1:]):
            res.append(s[i]+j)
    return res


def four_select_3(s):
    four_sring_list = perm(s)
    three_string_set = set()  # 初始化一个set
    for string in four_sring_list:
        for ind in range(len(string)):
            three_string = string[: ind] + string[ind + 1:]
            three_string_set.add(three_string)
    return three_string_set

s = 'ABCD'
res = []
fore = four_select_3(s)
later = four_select_3(s)
for ss in fore:
    for jj in later:
        six = ss + jj
        if six[0] != six[3] and six[1] != six[4] and six[2] != six[5]:
            res.append(six)

print(len(res))









