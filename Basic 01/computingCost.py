def compute_cost(x,y,w,b):
    """
    x = data
    y = values
    w,b = parameters

    return cost of w,b parameter using linear regreation
    """

    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = x[i]*w + b
        cost = (f_wb-y[i])**2
        cost_sum += cost
    totalcost = (1 / (2 * m)) * cost_sum  
    return totalcost