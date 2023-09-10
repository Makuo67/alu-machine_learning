#!/usr/bin/env python3

"""FUnction that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """Join two arrays"""
    new_array = []
    for i in arr1:
        new_array.append(i)

    for j in arr2:
        new_array.append(j)

    return new_array


arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]
print(cat_arrays(arr1, arr2))
print(arr1)
print(arr2)
