# coding=utf-8
##### 牛客
## 直接用字典就可以了


s = 'aardvadrk'
dic = {}
res = ''
for i, char in enumerate(s):
    if char not in dic.keys():
        dic[char] = 1
        res += char
print(res)