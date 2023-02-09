# 已有一个random(n), 0的概率是1/n, 1的概率是2/n,   fun()
# 构造一个new_random(n), 0概率是1/2, 1也是1/2.    new_fun()

import random
def fun(n):
	temp = random.randint(0, n-1)
	res = 0
	if temp<=n:
		res += 1
	return res/n

fun(n) # 返回0的概率是1/n没错了: (temp=0); 返回1的概率是2/n: (temp=0,1)

def new_fun():
	res = fun(2)  # random.randint(0, 1): 0 or 1 概率各0.5
	if res == 0.5:
		return 0
	else:
		return 1 
