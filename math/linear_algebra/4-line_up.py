#!/usr/bin/env python3

"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays"""
    new_array = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        if isinstance(arr1[i], list) and len(arr1[i]) != len(arr2[i]):
            return None
        elif not isinstance(arr1[i], list):
            new_array.append(arr1[i] + arr2[i])
        elif isinstance(arr1[i], list):
            new_array.append([])
            for j in range(len(arr1[i])):

                new_array[i].append(arr1[i][j] + arr2[i][j])
    return new_array


# arr1 = [1, 2, 3, 4]
# arr2 = [5, 6, 7, 8]
# print(add_arrays(arr1, arr2))
# print(arr1)
# print(arr2)
# print(add_arrays(arr1, [1, 2, 3]))
