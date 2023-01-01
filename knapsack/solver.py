#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
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
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    if len(items) < 250:
        value, opt, taken = dynamic_programming(capacity,items)
    else:
        value, opt, taken = DF_BnB(capacity,items)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def estimated_value(itemIndex,items,new_id,capacity):
    """
    Maximum expected sackValue
    """
    sackWeight = 0
    sackValue = 0
    for item in items:
        if new_id[0,item.index]:
            if sackWeight + item.weight <= capacity:
                sackWeight += item.weight
                sackValue += item.value
            else:
                fraction = (capacity - sackWeight)/item.weight
                sackValue += fraction*item.value
                sackWeight = capacity
                break

    return sackValue

def DF_BnB(capacity,items):
    """
    Depth First Branch and Bound Search
    """
    for node in Node.instances:
        del node
    for i in range(len(Node.instances)):
        del Node.instances[0]
    Node.optimalSolution = None

    sorted_items = sorted(items,reverse=True,key= lambda x: x.value/x.weight)
    nodes2visit = []
    value_list = np.array([[item.value for item in items]],dtype=np.int32)
    estValue = estimated_value(-1,sorted_items,np.ones((1,len(items)),dtype=np.int8),capacity)
    initialNode = Node(capacity,0,estValue,-1,np.ones((1,len(items)),dtype=np.int8))

    children = Spawn(initialNode,items,sorted_items,value_list,capacity)
    nodes2visit = children + nodes2visit

    while nodes2visit:

        children = []
        currentNode = nodes2visit.pop(0)

        if Node.optimalSolution:
            if currentNode.level == len(items) - 1 and currentNode.value >= Node.optimalSolution.value:
                    Node.optimalSolution = currentNode
            elif currentNode.estimated_value >= Node.optimalSolution.value:
                children = Spawn(currentNode,items,sorted_items,value_list,capacity)
            else:
                continue
        else:
            if currentNode.level == len(items) - 1:
                Node.optimalSolution = currentNode
            else:
                children = Spawn(currentNode,items,sorted_items,value_list,capacity)
                    
        nodes2visit = children + nodes2visit

    return int(Node.optimalSolution.value), 1, Node.optimalSolution.id[0]

class Node():

    optimalSolution = None
    instances = []

    def __init__(self,capacity,value,estimated_value,level,id_list):
        self.capacity = capacity
        self.value = value
        self.estimated_value = estimated_value
        self.level = level
        self.id = id_list
        Node.instances.append(self)

    def __str__(self):
         return f'Level: {self.level}, Value: {self.value}, Est Val: {self.estimated_value}, Cap: {self.capacity}'

def Spawn(node,items,sorted_items,value_list,capacity):
    '''
    Spawn children of the nodes for the DF BnB
    '''
    children = []

    itemIndex = node.level + 1

    if itemIndex == len(sorted_items):
        return []
    
    for i in reversed(range(2)):
        new_capacity = node.capacity - sorted_items[itemIndex].weight * i
        if new_capacity >= 0: # Feasible solution
            new_id = node.id.copy()
            new_id[0,sorted_items[itemIndex].index] = i
            if i == 0:
                new_est_val = estimated_value(itemIndex,sorted_items,new_id,capacity)
            else:
                new_est_val = node.estimated_value
            
            new_value = node.value + items[itemIndex].value * i
            child = Node(new_capacity,new_value,new_est_val,itemIndex,new_id)
            if Node.optimalSolution:
                if child.estimated_value > Node.optimalSolution.value:
                    children.append(child)
            else:
                children.append(child)

    return children

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
    matrix = np.zeros((capacity+1,len(items)+1),dtype=np.int32)
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

