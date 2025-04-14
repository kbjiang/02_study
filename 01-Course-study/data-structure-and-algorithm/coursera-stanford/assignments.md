### 1.3 count number of comparison in QuickSort
```python
def swap(alist, i, j):
    _ = alist[i]
    alist[i] = alist[j]
    alist[j] = _
    return alist

def partition(alist, idl, idr):
    # print(alist[idl:idr], idl, idr)
    # print(alist)
    p = alist[idl]
    i = idl + 1
    for j in range(idl+1, idr):
       if alist[j] < p:
            alist = swap(alist, i, j)
            i += 1
    alist = swap(alist, idl, i-1)
    # return i-1
    return i-1, idr - idl - 1

def choose_1st_as_pivot(alist, idl, idr):
    return idl

def choose_lst_as_pivot(alist, idl, idr):
    return idr - 1

def choose_m3_as_pivot(alist, idl, idr):
    idm = (idr - 1 + idl)//2
    tups = [(idx, alist[idx]) for idx in [idl, idm, idr-1]]
    tups = sorted(tups, key = lambda x: x[1])
    # print(tups)
    return list(zip(*tups))[0][1]

def quick_sort(alist, idl, idr):
    if idl >= idr - 1:
        return 0
    # i = choose_m3_as_pivot(alist, idl, idr)
    # i = choose_1st_as_pivot(alist, idl, idr)
    i = choose_m3_as_pivot(alist, idl, idr)
    alist = swap(alist, idl, i)
    # k = partition(alist, idl, idr)
    # quick_sort(alist, idl, k)
    # quick_sort(alist, k+1, idr)
    k, n_comp = partition(alist, idl, idr)
    n_comp += quick_sort(alist, idl, k)
    n_comp += quick_sort(alist, k+1, idr)
    return n_comp
```