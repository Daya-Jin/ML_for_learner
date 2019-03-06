# 欧氏距离
def E_dist(a:list,b:list):
    a=np.array(a)
    b=np.array(b)
    return np.linalg.norm(a-b)