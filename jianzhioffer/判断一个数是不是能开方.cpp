# 判断一个数是不是能开方 不能调用函数
# https://blog.csdn.net/qq_19446965/article/details/81951509

// 方法3：复杂度为O(n^0.5)。方法3中利用加减法替换掉了方法1中的乘法，所以速度会更快些。
    public static boolean isSquare(int num) {
        int temp = 1;
        for (int i = 1; i < num; i++) {
            num -= temp;
            if (num < 0)
                return false;
            else if (num == 0) {
                return true;
            }
            temp += 2;
        }
        return false;
    }




// 二分法实现
def mySqrt(self, x):
    l, r = 0, x
    while l <= r:
        mid = (l+r)//2
        if mid * mid <= x < (mid+1)*(mid+1):
            return mid
        elif x < mid * mid:
            r = mid
        else:
            l = mid + 1