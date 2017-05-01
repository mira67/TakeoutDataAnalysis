# -*- coding: utf-8 -*-

f='E:/myprojects/takeout/code/python/pattern_shoplist.csv'

file = open(f, 'r') 
#read lines in a list
all_shop_lines = file.readlines()
#for each line, each shop can be splited using ',' for further indexing
for i in range(0,900):
    one_line_of_shops = all_shop_lines[i].split(',')
    print i, len(one_line_of_shops)

