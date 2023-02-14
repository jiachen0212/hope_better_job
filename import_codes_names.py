# coding=utf-8
lines = open('./code.py').readlines()
lines = [a for a in lines if a[:4]=='def ']
for line in lines:
    print(line) 