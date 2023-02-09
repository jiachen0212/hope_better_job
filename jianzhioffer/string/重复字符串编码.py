# coding=utf-8
# 重复字符串编码

# 我自己写的，巨复杂....完全没必要这样....  虽然也是对的.....
# def fun(strs):
#     ll = len(strs)
#     count = [1]
#     chars = [strs[0]]
#     if strs[0] == strs[1]:   # 处理第一个字符的情况
#         count[-1] += 1
#     for i in range(1, ll - 1):
#         if strs[i] == strs[i + 1]:
#             if strs[i - 1] != strs[i]:   # 证明是重新扫到了新的重复元素
#                 count.append(1)
#                 chars.append(strs[i])
#             count[-1] += 1
#         elif strs[i - 1] != strs[i]:   # 和前一个也不等，和后一个也不等，那就是只出现一次了
#             count.append(1)
#             chars.append(strs[i])
#     if strs[-1] != strs[-2]:  # 处理最后一个字符的情况
#         count.append(1)
#         chars.append(strs[-1])
#     res = ''
#     for ind, num in enumerate(count):
#         res += (str(count[ind]) + chars[ind])
#     return res

# res = fun('BRAABCCDAAFTR')
# print(res)



#### 别人 ac 的
ss = list('BRAABCCDAAFTR')
p = []  # 两个2个的存，先存数值再存字符
while ss:
    a = ss.pop(0)  # 当前的字符
    if p == []:
        p.append('1')
        p.append(a)
        continue
    elif a == p[-1]:
        b = str(int(p.pop(-2)) + 1)  # b 更新计数值
        p.insert(-1,b)   # 这里是-1位置插入，特别注意！！！
        continue
    else:  # 不想等，直接就计数是1.然后加上这个新的字符
        p.append('1')
        p.append(a)
print ''.join(p)



##### 别人 ac 2
s = list('BRAABCCDAAFTR')
ss=''
temp=s[0]
count=1
for i in range(1,len(s)):
    if s[i]==temp:
        count+=1
    else:
        ss+=str(count)+temp
        temp=s[i]
        count=1
ss+=str(count)+temp   # 最后加这句是因为最后那个字符，temp和count虽然更新了，但是并没有加进ss中
print(ss)

