from legacy_corr import calculate

n, mean_x, mean_y, r = calculate([1, 2, 3], [2, 4, 6])
assert abs(r - 1.0) < 1e-6
print("测试结果：", r)