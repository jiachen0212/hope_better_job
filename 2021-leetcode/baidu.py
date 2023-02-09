# coding=utf-8
import time
from functools import wraps

# 1. 装饰器
def fn_timer(function):
    '''
    函数运行计时打印函数，装饰器形式
    :param function: 需要计时的函数
    :return:
    '''

    @wraps(function)  # 这里对应的function就是main函数了..
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0)))
        return result

    return function_timer

# 利用装饰器，完成计时功能
@fn_timer
def main(file_path, load_vec):
    '''
    some code
    '''



# 2. yield和return
'''
yield 针对list，循环等数据，可依次一一返回数据，
而return只能在循环结束后一次性返回所有结果.

'''
# return yield混用 test
class TestYield:
    def gen_iterator(self):
        for j in range(3):
            print("do_something-{}".format(j))
            yield j

    def gen_iterator_middle(self):
        print("gen_iterator_middle")
        # 返回的是迭代器的句柄，所以加一层return不影响是可以理解的
        return self.gen_iterator()

    def call_gen_iterator(self):
        # yield并不是直接返回[0,1,2]，执行下边这句后result_list什么值都没有
        result_list = self.gen_iterator_middle()
        # i每请求一个数据，才会触发gen_iterator生成一个数据
        for i in result_list:
            print("call_gen_iterator-{}".format(i))

'''
gen_iterator_middle
do_something-0
call_gen_iterator-0
do_something-1
call_gen_iterator-1
do_something-2
call_gen_iterator-2

'''
if __name__ == "__main__":
    obj = TestYield()
    obj.call_gen_iterator()




# 3. linux 基础
# nohup python3 __main__.py >log.log 2>&1 &  后台挂命令并写入log
# awk: linux处理文本的工具
'''
txt_cc:
fuwh:men:jiangxi
zhangsan:girl:shanghai
lisi:girl:beijing
wangwu:feman:shenzhen

awk 'BEGIN{FS=":"} {if($2=="girl") x++} END{print "女生个数:" x}' txt_cc
女生个数:2
awk 'BEGIN{FS=":"}{if($1=="fuwh") print(length($3))}' txt_cc
7

'''



# 4. python 基础
# python 多线程进程
# GIL: 全局解释器锁(也并不可保证百分百数据安全)
'''
一个线程有两种情况会释放GIL:
1. 线程进入IO操作之前，会主动释放. 所以io密集型任务适合使用多线程.
2. 解释器不间断的运行了 1000字节码(py2)或 15毫秒(py3) 后，会释放GIL

'''
# 多线程: threading   cpu io密集场景下宜使用   m个cpu, 线程数可: m+1、2m、io操作很多的话可设10m
# 多进程: from multiprocessing import Process  计算/cpu密集型场景



# python23区别: py2中的xrange在py3中合并为range,处理大量数据时可迭代器式产出，内存更友好
# py3增加一个 nonlocal 修饰词，实现非局部变量声明(比如c出了foo函数，但这个c变量还可以被访问到)
def func():
    c = 1
    def foo():
        nonlocal c # 注释掉这句的话，c=1
        c = 12
    foo()
    print(c)
func()  # c=12


# tuples and list
# 元组和列表的区别
# 元组不可变，列表可以，其他其实没啥区别
# 元组初始化更快，占用内存更小，但没列表灵活
# 字典dict的key，不可以是可变对象!
# 元组内的每个元素可以是不同类型的，如:(66, [1,6,9], )  + 实现元组扩增




# 大数据相关:
# MapReduce: Input, Split, Map, Shuffle, Reduce, Finalize.
# MapReduce的倒排索引? 倒排索引就是定位每个单词(文档)出现在哪些文档里，即可快速检索出这些单词.

# hadoop: map->shuffle->reduce
# https://zhuanlan.zhihu.com/p/20176725
'''
海量的数据，无法放在同一台机器中处理完(内存有限),怎么办呢?

1. map: 首先数据分摊到各个机器中(如1-5号机器)，然后每个机器针对自己的数据，做好map(key-value)
如搜索关键词数据，每台机器做好自己本地工作: 有哪些关键词，分别出现了多少次

2. shuffle: 用另外几台机器(A-E)，做以上几台机器中的，不同关键词数量统计:
机器A统计1-5机器中关键字key1、key2(或更多，视内存定)的次数，
机器B统计1-5机器中关键字key3、key4 的次数，
...

3. 最后的reduce，就是根据2中的shuffle结果，整理出总数据的各个key-value结果啦~

'''



# 5. 概率题
# Q: 如果明天下午的概率是80%，然后明天每个小时下雨的概率都是相等的， 求明天0点到8点的不下雨的概率（有一个小时下雨就算下雨）
# A: 0~8、8~16、16~24 时间段下雨应该是等概率的，为sqrt(1-0.8, 3)[0.2开三次方]
# 所以 res = sqrt(0.2, 3)




# 6. 工程算法题
# 外部排序 (数据很大10T，内存小4T，如何实现大数据排列有序)
# https://www.cnblogs.com/johnsblog/p/3943352.html
# 多路归并排序法:
# 先数据分成若干小段，依次在内存中排序好，然后再进行多路(k路)归并排序
# 失败树找k个数中的最小，时复为logK，不再是K-1.
# 因此,总数据n，分为m份，使用K路归并，时间复杂度为log m* (n-1)



# 7. 数据不平衡怎么处理? 欠采样、重采样、dataload类别级均衡采样、loss/weights惩罚/加权、【SMOTE算法】.
'''
SMOTE算法: 对少数量类别样本进行分析和模拟，将人工模拟的新样本添加到数据集中，起到补充数据作用。
KNN技术：

　　采样最邻近算法，计算出每个少数类样本的K个近邻;
　　从K个近邻中随机挑选N个样本进行随机线性插值;
　　构造新的少数类样本，将新样本与原数据合成，产生新的训练集;
https://blog.csdn.net/qq_33472765/article/details/87891320

交叉验证: train三等分 12test3 13test2 23test1 总和3次模型的结果均值作为最终模型效果.
可使得到更稳定的算法模型!!
'''



# 8. 如何快速的求1亿张图像中相似的图片?  获取图像的hash值~
# A: 对图片先进行hash，再比较两张图片hash值的汉明距离。至于hash的办法有很多了…比如【算小波特征】啊啥的...
# Hash算法有三种: 平均哈希算法(aHash)、感知哈希算法(pHash)、差异哈希算法(dHash)
# https://www.cnblogs.com/Kalafinaian/p/11260808.html



# 9. DL相关
# 空洞卷积的优缺点:
'''
1. 局部信息丢失，多个空洞卷积叠加，因为kernel并不是连续的，因此会引起图片中并不是所有的像素点都用来计算，损失了信息的连续性。

2. 远距离信息的相关性缺失，大rate的空洞卷积只对大尺寸的物体分割有用，对小物体来说没有什么用处。


SSD检测小目标差
Q: 浅层feature map语义不足，虽然map尺寸较大，但不足以检测定位出小目标的特征
深层feature map语义信息足够，但map尺寸太小，且一个pixel可能"涵盖"有重叠的几个大小目标，导致特征混淆，又不利于小目标的定位识别了..
加个fpn结构，或者两阶段rpn，可能可以辅助点小目标识别!

再就是小目标对象，本来就比较难被很多anchor给"覆盖"到.(就是因为它小，它的gt没有和很多的anchor产生>thres的情况，
所以天然的，小目标就没能被网络好好的看到专心学好，导致小目标识别+定位差..)

为什么SSD的data augmentation能涨这么多点，就是因为通过randomly crop，让每一个anchor都得到充分训练（也就是说，crop出一个小物体，在新图里面就变成大物体了）

'''
