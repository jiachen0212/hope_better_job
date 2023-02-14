
def solve(a, b):
    a = sorted(a)
    b = sorted(b)
    ans = 0
    while len(a):
        if a[-1] > b[-1]:  # 田忌最快的马比齐威王最快的马 快，两者比赛，田忌赢
            ans += 200
            del a[-1], b[-1]
        elif a[-1] < b[-1]:  # 田忌最快的马比齐威王最快的马 慢， 田忌输
            ans -= 200
            # 选择田忌最慢的马比齐威王最快的马比赛
            del a[0], b[-1]
        else:  # 田忌最快的马和齐威王最快的马一样快时
            if a[0] > b[0]:  # 田忌最慢的马比齐威王最慢的马快，田忌赢
                ans += 200
                del a[0], b[0]
            else:
                if a[0] < b[-1]:  # 田忌最慢的比齐王最快慢
                    ans -= 200
    return ans 
print('田忌赛马: ', solve([92,83,71],[95,87,74]))