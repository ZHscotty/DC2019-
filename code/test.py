def solution(n, m):
a = []
f = []
z = []
for x in range(n):
    a.append(x+1)
group = n/m
start = 0
for x in range(group):
    end = start + (m-1)
    b = a[start:end]
    sum_temp = sum(b)
    if x%2 == 0:
        f.append(sum_temp)
    else:
        z.append(sum_temp)
    start = end
fu = sum(f)
zheng = sum(z)
return zheng-fu

return n, m