#coding=utf-8
# 剪绳子问题: 分别动态规划和贪婪算法实现

def dtgh(a):
    if a < 2:
        return 0
    if a == 2:
        return 1
    if a == 3:
        return 2
    products = [0, 1, 2, 3]  # products存储子问题的最优解,需要的时候直接调用,省去多次重复计算
    for i in range(4, a + 1):
        curpro = 0
        for j in range(1, i / 2 + 1):  # 把每一个i值下的分段乘积最大值存在products里.
            if products[j] * products[i - j] > curpro:
                curpro = products[j] * products[i - j]
        products.append(curpro)
    # print products
    print products[-1], 'dtgh'
    return products[-1]


# 贪婪算法就是当a>5时,尽量多分出3,如果剩余4,就2/2.
def tanlan(a):
    if a < 2:
        return 0
    if a == 2:
        return 1
    if a == 3:
        return 2
    timeof3 = a / 3
    ys = a % 3
    if ys == 1:
        timeof3 -= 1
    timeof2 = (a - timeof3 * 3) / 2  # 如果余数是1, timeof3减了1，所以timeof2变成了2
    # 余数是2的话, timeof2则为1.
    print pow(3, timeof3) * pow(2, timeof2), 'tanlan'
    return pow(3, timeof3) * pow(2, timeof2)

tanlan(16)
dtgh(16)