# coding=utf-8
'''
依图一面code
01串，两次删除，每次只能删连续且一致的子串，共可以删多少个字符串？

'''

class Solution():
    """"""
    def count(self, s):
        num_counter = []
        i = 0
        while i<len(s):
            count = 1
            j=i+1
            while j<len(s) and s[j]==s[i]:
                count +=1
                j +=1
            num_counter.append(count)
            i = j
        return num_counter

    def main(self, s):
        num_counter = self.count(s)
        if  len(num_counter) <= 3:
            return len(s)
        else:
            max_len = num_counter[-1]       # 存最大数
            sub_max_len = num_counter[-2]   # 存次大数
            # 因为下面的for循环遍历不到最后俩数，所以得把他们在init时安排上
            max_three_sum =0
            for i in range(len(num_counter)-2):
                # 求连续三数最大
                max_three_sum = max(num_counter[i] + num_counter[i+1] + num_counter[i+2], max_three_sum)
                # 求数组中最大和第二大的数
                if num_counter[i] > max_len:
                    sub_max_len = max_len
                    max_len = num_counter[i]
                elif num_counter[i] > sub_max_len:
                    sub_max_len = num_counter[i]
            # 连续三数 vs 最大+第二大
            return max(max_three_sum, max_len+sub_max_len)

if __name__ == '__main__':
    s = Solution()
    # res = s.main('000110011110001010000001000')
    # print(res)
    print s.main('1110111111111000000000')

