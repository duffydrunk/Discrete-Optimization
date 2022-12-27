#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        # items.append(Item(i-1, int(parts[0]), int(parts[1]),float(int(parts[0])/int(parts[1]))))
        items.append(Item(i-1, int(parts[0]), int(parts[1])))


    if len(items) < 250:
        value, opt, taken = dynamic_programming(capacity,items)
    else:
        value, opt, taken = greedy(capacity,items)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def greedy(capacity,items):
    
    taken = [0] * len(items)
    items.sort(key = lambda x : x.value/x.weight)
    sackWeight = 0
    sackValue = 0
    for item in items:
        if sackWeight + item.weight <= capacity:
            taken[item.index] = 1
            sackWeight += item.weight
            sackValue += item.value
    
    return int(sackValue), 0, taken


def dynamic_programming(capacity,items):
    taken = [0] * len(items)
    matrix = np.zeros((capacity+1,len(items)+1),dtype=int)
    for item in items:
        for c in range(capacity+1):
            if item.weight <= c:
                prev = matrix[c,item.index]
                current = matrix[c-item.weight,item.index] + item.value
                matrix[c,item.index+1] = max([prev,current])
            else:
                matrix[c,item.index+1] = matrix[c,item.index]

    current_weight = capacity 
    for i in range(1,len(items)+1):
        if matrix[current_weight,-i] == matrix[current_weight,-1-i]:
            continue
        else:
            taken[-i] = 1
            current_weight -= items[-i].weight

    return matrix[-1,-1], 1, taken

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

