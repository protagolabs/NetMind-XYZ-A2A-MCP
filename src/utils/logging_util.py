#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
日志工具类
"""

import logging

class PrefixFilter(logging.Filter):
    """为日志添加前缀的过滤器"""
    
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        if not hasattr(record, 'prefix_added'):
            record.msg = f"[{self.prefix}] {record.msg}"
            record.prefix_added = True
        return True 