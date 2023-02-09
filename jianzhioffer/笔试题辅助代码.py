# coding=utf-8

# 0~9999 中有多少个数字 5
count = 0
for i in range(9999):
    count += str(i).count('5')
print(count)


'''
# a, b 互换值
a^=b   # ^ 或运算
b^=a
a^=b
'''


# 十进制转Y进制
def NumToABC(intN):
    if intN in [0,1,2,3,4,5,6,7,8,9]:
        return intN
    if intN ==10 :
        return 'A'
    if intN ==11 :
        return 'B'
    if intN ==12 :
        return 'C'
    if intN ==13 :
        return 'D'
    if intN ==14 :
        return 'E'
    if intN ==15 :
        return 'F'
# 10进制转Y进制
def intToY(num, Y):
    res = ""
    while num:
        res = str(NumToABC(num%Y))+res
        num /= Y
    res.upper()
    return res

ans = intToY(12, 16)
print(ans, '----')




# coding=utf-8
import sys
def NumToABC(intN):
    if intN in [0,1,2,3,4,5,6,7,8,9]:
        return intN
def intToY(num, Y):
    res = ""
    while (num!=0):
        res = str(NumToABC(num%Y))+res
        num = num//Y #取商
    res.upper()
    return res

def fun(num, l):
    res = set()
    for i in range(num):
        binary = intToY(l[i], 2)
        ones = binary.count('1')
        res.add(ones)
    return len(list(res))
try:
    while True:
        bins = int(raw_input())
        res = []
        for i in range(bins):
            num  = int(raw_input())
            nums = sys.stdin.readline().strip()
            l = list(map(int, nums.split()))
            res.append(fun(num, l))
        for n in res:
            print n,
        break
except:
    pass




# 1*3的砖  铺成20*3的种数
f=[0]*21
f[1],f[2],f[3]=1,1,2
for i in range(4,21):
    f[i]=f[i-1]+f[i-3]
print(f[20], '---')   # 1278


# 连续抛硬币得到两个正面向上的平均次数
# sum(1/p^i)


# 掷骰子得到1～6的期望
e = [0]*7
e[1] = 1
# e[i+1]=e[i]+(6/6-i)
for i in range(1,6):
    e[i+1]=e[i]+ round(6/(6-i))
print(e)   # 14 手算出来是14.7

# 另一个公式版计算：
s = 0
for i in range(1,7):
    s += round(6/i)
print s   # 一样也是14

