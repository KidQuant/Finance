
n = 241

def sum_odd_squares(n):
    sum = 0
    for n in range(n-1,0,-1):
        if n % 2 == 1:
            sum += n*n
    return sum

print(sum_odd_squares(n))

