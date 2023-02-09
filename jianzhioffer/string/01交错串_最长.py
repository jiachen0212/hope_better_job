# coding=utf-8
####### 牛客

string = raw_input()
res, tmp_max = 1, 1   # res统计一路扫过来的max长度  tmp_max是当前01串的长度
for i in range(len(string) - 1):
    if string[i] != string[i + 1]:
        tmp_max += 1
        res = max(res, tmp_max)
    else: # 如果下个位置组不成01串，则从下个位置重新开始记录当前最大长度
        tmp_max = 1
print(res)

