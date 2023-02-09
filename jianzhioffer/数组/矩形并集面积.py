# coding=utf-8
# 求两举行的并集  area1+area2-重合

class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        return (C-A)*(D-B)+(H-F)*(G-E)\
        -max(0,min(C-E,G-A,G-E,C-A))*max(0,min(H-F,D-B,H-B,D-F))

s = Solution()
area = s.computeArea(-2,-2,2,2,3,3,4,4)
print area