# coding=utf-8

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
    res = ''
    while num:
        res = str(NumToABC(num%Y)) + res 
        num = num//Y   
    res.upper() # 转成大写 这行貌似可以不要
    return res

ans = intToY(12, 16)
print ans
# bug.....