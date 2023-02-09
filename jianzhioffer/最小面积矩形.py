# coding=utf-8
# leetcode
# https://leetcode-cn.com/problems/minimum-area-rectangle-ii/comments/
# 没思考  copy的code

class Solution:
    def minAreaFreeRect(self, points):
        res = 0  # 已知，所有的点都是不同的
        d = dict()  # 存储中点坐标相同的点序号列表，相同则可以进入下一步判断（使用坐标和代替中点坐标）
        num = len(points)
        for i in range(num-1):
            for j in range(i,num):
                x = points[i][0] + points[j][0]
                y = points[i][1] + points[j][1]
                if (x,y) not in d:
                    d[(x,y)] = []
                d[(x,y)].append((i,j))

        for key in d:
            if len(d[key])<2:
                continue
            dl = dict()  # 存储边长和边向量坐标
            for i,j in d[key]:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                l = dx*dx + dy*dy
                if l not in dl:
                    dl[l] = []
                dl[l].append((dx,dy))
            for l in dl:
                if len(dl[l])<2:
                    continue
                #出现矩形，对于圆的内接矩形而言，邻边长度差越大则面积越小，这里直接用对角向量叉积求取面积
                num = len(dl[l])
                for i in range(num-1):
                    for v2 in dl[l][i:]:
                        v1 = dl[l][i]
                        area = abs(v1[0]*v2[1]-v2[0]*v1[1]);  # 实际面积的两倍
                        if area:
                            if not res or area<res:
                                res = area
        if res:
            res /= 2.0
        return res
