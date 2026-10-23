from legacy_corr import calculate

n, mean_x, mean_y, r = calculate([1, 2, 3], [2, 4, 6])
assert abs(r - 1.0) < 1e-6

try:
    calculate([], [])
except ValueError:
    pass
else:
    raise AssertionError("空数据应当报错")

try:
    calculate([1, 1, 1], [2, 3, 4])
except ValueError:
    pass
else:
    raise AssertionError("常量列应当报错")
n, mean_x, mean_y, r = calculate([1, 2, 3], [6, 4, 2])
assert abs(r + 1.0) < 1e-6
print("测试通过")