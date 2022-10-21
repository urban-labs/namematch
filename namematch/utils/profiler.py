import os

from functools import wraps, partial

from line_profiler import LineProfiler


class Profiler:

    line_profiler = LineProfiler()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args[0].enable_lprof:
                self.line_profiler.add_function(func)
                self.line_profiler.enable_by_count()
            return func(*args, **kwargs)
        return wrapper

