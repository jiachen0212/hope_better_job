# coding=utf-8
dic = {"c": 3, "h": 4, "e":5, "n": 6, "j": 0, "i": 1, "a": 2}
# sort base value

def sort_by_value(dic):
    items = dic.items()
    backitems = [[v[1], v[0]] for v in items]  # 把value先放,key后放
    backitems.sort()  #默认按首位置元素排序,也即value排序
    return [backitems[i][1] for i in range(len(backitems))]  # 打印排序后的value
    # return [backitems[i][0] for i in range(len(backitems))]  # 打印排序后的key
    # return [backitems[i][:] for i in range(len(backitems))]  # 打印排序后的key和value


res = sort_by_value(dic)
print res

print [v for v in sorted(dic.values())]  # 仅排序value,没带上key


