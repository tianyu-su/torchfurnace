# -*- coding: utf-8 -*-
# Date: 2020/3/27 16:27

"""
module description
"""

__author__ = 'tianyu'

import functools


def test_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'\n\nTesting function start: "{func.__name__}"')
        ret = func(*args, **kwargs)
        print(f'Testing function done: "{func.__name__}"')
        return ret

    return wrapper
