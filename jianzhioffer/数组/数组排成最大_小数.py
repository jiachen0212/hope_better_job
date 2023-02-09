# coding=utf-8

###### 牛客 ac

# 将数组中的元素排列组合成最小数
'''
将数组中的数字转换成string进行操作比较，比较规定如下
任意两个str1 和str2进行比较
首先连接成
num1 = str1+str2
num2 = str2+str1
转换成int型
① intnum1 > intnum2 str1>str2
② intnum1 < intnum2 str2>str1
③ intnum1==intnum2 str1=str2
'''
# 重点思路就是把int转换成str处理的思想
class Solution:
    def PrintMinNumber(self, numbers):
        if not numbers:
            return ''
        numstr = map(str,numbers)    # [1,2,3] ==> ['1, '2', '3']
        l = lambda n1,n2: int(n1+n2)-int(n2+n1)   # 指定比较方式   小于0

        # l = lambda n1,n2: int(n2+n1)-int(n1+n2)  # 最大

        numsort = sorted(numstr,cmp=l) #用特定的比较方式cmp进行排序
        # 排完序之后用join进行连接成结果
        # str(int("".join(i for i in numsort)))  是为了防止"00"这种情况
        return str(int("".join(i for i in numsort)))


s = Solution()
res = s.PrintMinNumber([9, 32, 921])
print(res)