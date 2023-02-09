def fun1(s):
    l = len(s)
    val = []   # 存放不同的数字
    time = []  # 存放出现的次数
    for i in range(l):
        if s[i] not in val:  # 判断一个元素是否在list中
            val.append(s[i])
            time.append(1)   # time里添加一个计数位置
        else:
            ind = val.index(s[i])  # 查看这个重复出现的元素的索引
            time[ind] += 1  # 在对应次数位置上+1
    maxtime = max(time)
    if maxtime < l >> 1:  # 最大出现次数小于数字长的一半
        return False
    res = val[time.index(maxtime)]
    return res