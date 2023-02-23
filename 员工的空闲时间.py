class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        rec = []
        for sche in schedule:
            for inter in sche:
                s = inter.start
                e = inter.end
                rec.append([s, e])
                
        rec.sort()
        res = []
        pre = rec[0][1]
        for s, e in rec:
            if pre < s:
                res.append(Interval(pre, s))
            pre = max(pre, e)
        return res
