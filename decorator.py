# https://zhuanlan.zhihu.com/p/269012332
from time import time, sleep


#1. 无参数的装饰器
def run_time(func):
    # 计时函数wrapper()
    def wrapper():
        start = time()
        func()                  # 需要计时的函数:func()
        end = time()
        cost_time = end - start
        print("func cost: {}s".format(cost_time))
    return wrapper
 
@run_time
def fun_one():
    sleep(1)


# @run_time装饰器, 给fun_one()函数完成计时功能.
# @run_time可给任何函数, 完成计时功能, 无需再重复写wrapper()代码.
print('1. 无参数的装饰器:')
fun_one()

print('\n')

print('2. "有"参数的装饰器:')
# log功能装饰器, 也就是func可自主设参.
# 装饰器的参数, 只可能是函数
# 传给内嵌真是运行fun()的参数, 写为wrapper(*args, **kw), fun(*args, **kw). 
# 装饰器一定要, return wrapper
def logger(func):
    def wrapper(*args, **kw):
        print('主人, 我准备开始执行: {} 函数了:'.format(func.__name__))
        func(*args, **kw)
        print('主人, 我执行完啦.')
    return wrapper

@logger
def func(a,b):
    print("{}+{}={}".format(a,b,a+b)) 

# 运行带有装饰器的func()
func(100, 20)


print('\n')
# 给装饰器传参数
print('3. 给装饰器传参数')

# 装饰器函数传入, 国籍参数
# 两级嵌套
def say_hello(contry):
    def wrapper(func):
        def deco(*args, **kwargs):
            if contry == "china":
                print("你好!")
            elif contry == "america":
                print('hello~')
            else:
                return
            func(*args, **kwargs)
        return deco
    return wrapper

# 小明, 中国人
@say_hello("china")
def xiaoming(a, b):
    print("{}+{}={}".format(a,b,a+b)) 

# jack，美国人
@say_hello("america")
def jack(a, b):
    print("{}+{}={}".format(a,b,a+b)) 

# 运行func(), func的参数, *args, **kwargs传入了
# 装饰器say_hello(contry)也执行了参数.
xiaoming(1,2)
jack(3,4)


# 4. 
print('\n4. 不带参数的类装饰器')
# 装饰器是一个类, 必须实现: __init__, __call__这俩函数
class logger(object):
    # __init__还是传入func函数作为参数
    def __init__(self, func):
        self.func = func

    # __call__内层传入给func的参数
    def __call__(self, *args, **kwargs):
        print("[INFO]: the function {func}() is running..."\
            .format(func=self.func.__name__))
        return self.func(*args, **kwargs)

@logger
def say(something):
    print("say {}!".format(something))

# 运行带装饰器的函数.
say("chenjia")


# 5. 
print('\n5. 带参数的类装饰器')
# __init__: 不再接收被装饰函数, 而是接收传入参数
# __call__: 接收被装饰函数，实现装饰逻辑
class logger(object):
    def __init__(self, level='INFO'):
        self.level = level

    def __call__(self, func): # 接受函数
        def wrapper(*args, **kwargs):
            print("[{level}]: the function {func}() is running..."\
                .format(level=self.level, func=func.__name__))
            func(*args, **kwargs)
        return wrapper  

@logger(level='WARNING')
def say(something):
    print("say {}!".format(something))

say("chenjia")


# 6.
print('\n6. 使用偏函数与类实现装饰器')
# 只要实现了__call__, 就可被作为装饰器类
import time
import functools

class DelayFunc:
    def __init__(self,  duration, func):
        self.duration = duration
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f'Wait for {self.duration} seconds...')
        time.sleep(self.duration)
        return self.func(*args, **kwargs)

    def eager_call(self, *args, **kwargs):
        print('Call without delay')
        return self.func(*args, **kwargs)

def delay(duration):
    """
    装饰器：推迟某个函数的执行。
    同时提供 .eager_call 方法立即执行
    """
    # 此处为了避免定义额外函数，
    # 直接使用 functools.partial 帮助构造 DelayFunc 实例
    return functools.partial(DelayFunc, duration)

@delay(duration=2)
def add(a, b):
    return a+b

print(add)
print(add(2,3))
print(add.func)









