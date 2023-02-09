# coding=utf-8
# 区间插入
# https://www.cnblogs.com/grandyang/p/4367569.html


bins = [[1,2],[3,5],[6,7],[8,10],[12,16]]
new_bin = [4,8]
begs = []
ends = []

for bi in bins:
    begs.append(bi[0])
    ends.append(bi[1])

# 这两行可以不要，给的就是有序的区间
# begs.sort()
# ends.sort()

l = len(begs)
cur = 0
res = []
for i in range(l):
    if ends[i] < new_bin[0]:
        res.append([begs[i], ends[i]])
        cur += 1
    elif begs[i] > new_bin[1]:
        res.append([begs[i], ends[i]])  # 注意这里没有cur+1
        # cur是用来记录更新的bin放在哪个位置
    else:  # 插入区间的精髓
        new_bin[0] = min(new_bin[0], begs[i])
        new_bin[1] = max(new_bin[1], ends[i])
res.insert(cur, [new_bin[0], new_bin[1]])
print(res)