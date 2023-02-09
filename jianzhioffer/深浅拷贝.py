# coding=utf-8
# 自己实现深拷贝和浅拷贝

# 递归实现深拷贝   区分dict tuple list 等进行拷贝
# 转成各种数据结构的基层元素，一一拷贝

def deepcopy(cls):
    if isinstance(cls, dict):
        return {k: deepcopy(v) for k, v in cls.items()}
    elif isinstance(cls, list):
        return [deepcopy(item) for item in cls]
    elif isinstance(cls, tuple):
        return tuple([deepcopy(item) for item in cls])
    else:
        return cls
