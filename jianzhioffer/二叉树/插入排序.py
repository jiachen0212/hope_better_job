# coding=utf-8

countLIst = [9,1,44,23,123,77,312,323,53]
for j in range(1,len(countLIst)):
    key = countLIst[j]
    i = j - 1
    while i>=0 and countLIst[i]>key:
        countLIst[i+1] = countLIst[i]
        i = i-1
    countLIst[i+1]=key
print(countLIst)
