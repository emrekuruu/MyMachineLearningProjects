import numpy as np

x = []
y = []
alfa = 0.001
costHist = []

for i in range(1, 50):
    number = i
    x.append(number)
    y.append(number * 100)

m = len(x)

x_train = np.asarray(x)
y_train = np.asarray(y)


def costFunc(w, b, x, y):
    total_cost = 0
    for i in range(m):
        model = w * x[i] + b
        total_cost += (((model - y[i]) ** 2) * x[i]) // 2 * m
        return total_cost


def gradient(w, b, x, y):
    temp_w = 0
    temp_b = 0

    for i in range(m):
        model = w * x[i] + b
        temp_w += w - alfa * (model - y[i]) * x[i]
        temp_b += b - alfa * (model - y[i])

    w = temp_w / m
    b = temp_b / m

    return w, b


def myLoop(w, b, x, y):
    iteration = 0
    while (iteration < 100000):
        cost = costFunc(w, b, x, y)
        costHist.append(cost)
        print(f"iteration {iteration} cost {cost}")
        w, b = gradient(w, b, x, y)
        iteration += 1
        if (cost == 0):
            print(f"\nw is {w} and b is {b} ")
            return w, b
            break


print("Enter the size of your house in 1000m^2")
size = input()
w, b = myLoop(2, 5, x_train, y_train)

print(f"prediction : the function is {w} x+{b} \n Total = " + str(w * float(size) + b))
