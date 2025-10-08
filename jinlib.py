import math
import random
from random import randint
# use a list to represent a polynomial

def polySum(f, g):
    # f,g: lists, i.e., polynomials f(s) and g(s)
    # return h(s)=f(s)+g(s)
    N = len(f)
    M = len(g)
    K = max(N, M)
    h = [0] * K
    for i in range(N):
        h[K - 1 - i] = h[K - 1 - i] + f[N - 1 - i]
    for i in range(M):
        h[K - 1 - i] = h[K - 1 - i] + g[M - 1 - i]
    if len(h) > 1:
        while h[0] == 0 and len(h) > 1:
            del h[0]
    return h


# -----------------------------------------

def polyTimes(f, g):
    # f, g: lists, i.e., polynomials f(s), g(s)
    # return h(s)=f(s)*g(s)
    # exercise: complete this function
    N = len(f)
    M = len(g)
    K = N + M - 1
    h = [0] * K
    for i in range(N):
        for j in range(M):
            h[i + j] = h[i + j] + f[i] * g[j]
    if len(h) > 1:
        while h[0] == 0 and len(h) > 1:
            del h[0]
    if len(h) == 0:
        h = [0]
    return h


# ---------------------------------
'''
p1=[1,2,3,2,1]
p2=[1]
p3=[2,3,-1]
print(p1,p2,p3)
print(polyTimes(p1,p2),polyTimes(p1,p3))
p2=[0]
p3=[]
print(p1,p2,p3)
print(polyTimes(p1,p2),polyTimes(p1,p3),polyTimes(p3,p3))
'''


# ----------------------------------------

def polyScale(k, f):
    # k: number. f: list, i.e., polynomial f(s)
    # return h(s)=k*f(s)
    # exercise: complete this function
    N = len(f)
    h = []
    for i in range(N):
        h = h + [k * f[i]]

    if len(h) > 1:
        while h[0] == 0 and len(h) > 1:
            del h[0]
    if len(h) == 0:  # if h is empty
        h = [0]
    return h


# --------------------------
'''
p=[1,2,3,2,1]
k=3.33
print(k,p,polyScale(k,p))
k=0
print(k,p,polyScale(k,p))
p=[0]
k=7
print(k,p,polyScale(k,p))
p=[]
k=10
print(k,p,polyScale(k,p))
p=[]
k=0
print(k,p,polyScale(k,p))
'''


# --------------------------


def polyDivid(f, g):
    # f, g: lists, i.e., polynomials f(s), g(s)
    # return u(s) and v(s), f(s)/g(s)=u(s)+v(s)/g(s), order(v(s))<order(g(s))
    # or, f(s)=u(s)*g(s)+v(s).
    # exercise: complete this function

    # remove first few zeros from g
    gg = []
    for i in range(len(g)):
        gg = gg + [g[i]]
    if len(gg) > 1:
        while gg[0] == 0 and len(gg) > 1:
            del gg[0]

    if len(gg) == 0 or gg[0] == 0:  # quit with error
        tk = 6 / 0
    else:
        N = len(f)
        M = len(gg)
        K = N - M + 1
        u = [0] * K

        # copy f into w
        w = []
        for i in range(N):
            w = w + [f[i]]

        # long division
        for i in range(K):
            u[i] = w[i] / gg[0]
            for j in range(M):
                w[i + j] = w[i + j] - u[i] * gg[j]
        v = w

        if len(u) > 1:
            while u[0] == 0 and len(u) > 1:
                del u[0]
        if len(v) > 1:
            while v[0] == 0 and len(v) > 1:
                del v[0]
        if len(u) == 0:
            u = [0]
        if len(v) == 0:
            v = [0]
        return [u, v]


# -------------------------

'''
a=[2,3,4,5,6]
b=[1,1]
print(a,b,polyDivid(a,b))
a=[2,3,4]
b=[1,1,2,3]
print(a,b,polyDivid(a,b))
a=[2,3,4,0]
b=[3,1,2,3]
print(a,b,polyDivid(a,b))

a=[]
b=[3,1,2,3]
print(a,b,polyDivid(a,b))

a=[0]
b=[3,1,2,3]
print(a,b,polyDivid(a,b))

a=[0,0,1,2,-7,8]
b=[0,0,2,3]
print(a,b,polyDivid(a,b))

a=[2,3,4,0]
#b=[0] #or 
b=[]
print(a,b,polyDivid(a,b))
'''


# ----------------------


class matN:
    # matrix of numbers
    # basically, it is a list of lists
    def __init__(self, a):
        # a: list of list, list of rows, each row is also list
        N = len(a)
        if N == 0:
            print('error: Null matrix')
            2 / 0
        else:
            for i in range(N):
                if i == 0:
                    M = len(a[0])
                else:
                    if M != len(a[i]):
                        print('Wrong sizes')
                        2 / 0
                # print(a[i])
                for j in range(len(a[i])):
                    x = a[i][j] * 1.223
                    # print(f'value {x}') #an error will occur if a[i][j] is not a number
        self.height = N
        self.width = M
        self.ele = []
        for i in range(N):
            row = []
            for j in range(M):
                row = row + [a[i][j]]
            self.ele = self.ele + [row]
        # self.print()#<<----------------------------------------------

    def print(self):
        for i in range(self.height):
            print('  ', self.ele[i])
        print()

    # for A=B if B=matN(p), use A=matN(B.ele) instead.


# -------------------------------------------


'''
p=[[2,2],[3,2],[2,0]]
#p=[[2,2],[3,2,9],[2,0]]
#p=[[2,2],[3,2],[2]]
#p=[[2,2],[3+1j,2],[2,[9]]]
print(p)
A=matN(p)
A.print()
B=matN(A.ele)

for i in range(A.height):
    for j in range(A.width):
        A.ele[i][j]=i**2.3+j
print(p)
A.print()
B.print()
'''


# ---------------------------------------

class matS:
    # matrix of polynomials
    # basically, it is a list of lists of lists
    def __init__(self, a):
        N = len(a)
        if N == 0:
            print('error: Null matrix')
            2 / 0
        else:
            for i in range(N):
                if i == 0:
                    M = len(a[0])
                else:
                    if M != len(a[i]):
                        print('Wrong sizes')
                        2 / 0
                # print(a[i])
                for j in range(len(a[i])):
                    x = -1
                    for k in range(len(a[i][j])):
                        x = x + a[i][j][k] * 1.2
                    # print(f'value {x}') #an error will occur if a[i][j] is not a polynomial
        self.height = N
        self.width = M
        self.ele = []
        for i in range(N):
            row = []
            for j in range(M):
                K = len(a[i][j])
                poly = []
                for k in range(K):
                    poly = poly + [a[i][j][k]]
                row = row + [poly]
            self.ele = self.ele + [row]
        # self.print()#<<----------------------------------------------------

    def print(self):
        for i in range(self.height):
            print('  ', self.ele[i])
        print()

    # for A=B if B=matS(p), use A=matS(B.ele) instead.


# -------------------------------------------

'''

p=[[[2,2,4],[3,2,1]],[[2,0,-9],[3.8+9j]]]
#p=[[[2,2,4],[3,2,1]],[[2,0,-9],3.8+9j]]
A=matS(p)
B=matS(A.ele)
B.ele[0][0][0]=122+344j
A.print()
B.print()
'''


def matN2S(A):
    # A is a matN, return a matS
    w = []
    for i in range(A.height):
        row = []
        for j in range(A.width):
            row = row + [[A.ele[i][j]]]
        w = w + [row]
    B = matS(w)
    return B


'''
p=[[1,2,3],[2,3,1]]
A=matN(p)
B=matN2S(A)
A.print()
B.print()
'''


def matNeye(n):
    # return the unit matrix of size nxn
    if n < 1:
        print('matrix size must >0 ')
        2 / 0
    else:
        p = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row = row + [1]
                else:
                    row = row + [0]
            p = p + [row]
        C = matN(p)
        return (C)


# matNeye(5).print()

def matNzero(n, m):
    # return the zero matrix of size nxm
    if n < 1 or m < 1:
        print('matrix size must >0 ')
        2 / 0
    else:
        p = []
        for i in range(n):
            row = []
            for j in range(m):
                row = row + [0]
            p = p + [row]
        C = matN(p)
        return (C)


# A=matNzero(8,14)
# A.print()


def matSeye(n):
    # return the unit s-poly matrix of size nxn
    if n < 1:
        print('matrix size must >0 ')
        2 / 0
    else:
        p = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row = row + [[1]]
                else:
                    row = row + [[0]]
            p = p + [row]
        C = matS(p)
        return (C)


# matSeye(5).print()

def matSzero(n, m):
    # return the zero s-poly matrix of size nxm
    if n < 1 or m < 1:
        print('matrix size must >0 ')
        2 / 0
    else:
        p = []
        for i in range(n):
            row = []
            for j in range(m):
                row = row + [[0]]
            p = p + [row]
        C = matS(p)
        return (C)


'''
A=matSzero(8,14)
A.print()
'''


def matNSum(A, B):
    # A,B: matN
    # return C=A+B
    if A.height != B.height or A.width != B.width:
        print('error size')
        3 / 0
    else:
        p = []
        for i in range(A.height):
            row = []
            for j in range(B.width):
                row = row + [A.ele[i][j] + B.ele[i][j]]
            p = p + [row]
    C = matN(p)

    return C


'''
p=[[1,2,3],[2,1,3],[4,4,5]]
A=matN(p)
p=[[1,2,6],[2,-1,3],[4,2,-5]]
B=matN(p)
A.print()
B.print()
C=matNSum(A,B)
C.print()
'''


def matNScale(k, A):
    # k: number A:matN
    # return: k*A
    p = []
    for i in range(A.height):
        row = []
        for j in range(A.width):
            row = row + [k * A.ele[i][j]]
        p = p + [row]

    C = matN(p)

    return C


'''
p=[[1,2,3],[1,2,1],[3,-2,4]]
A=matN(p)
k=2.22
A.print()
print(k)
B=matNScale(k,A)
B.print()
'''


def matNTimes(A, B):
    # A,B: matN
    # return C=A*B

    if A.width != B.height:
        print('wrong sizes')
        5 / 0.0
    else:
        C = matNzero(A.height, B.width)
        for i in range(C.height):
            for j in range(C.width):
                Q = []
                for k in range(A.width):
                    Q = A.ele[i][k] * B.ele[k][j]
                    C.ele[i][j] = C.ele[i][j] + Q
        return (C)


'''
p=[[1,1,2,2],[1,0,2,1],[3,-2,1,2]]
A=matN(p)
A.print()
B=matNzero(4,5)
for i in range(B.height):
    B.ele[i][1]=1
B.print()
C=matNTimes(A,B)
C.print()
'''


def matSSum(A, B):
    # A,B: matS
    # return C=A+B
    if A.height != B.height or A.width != B.width:
        print('error size')
        3 / 0
    else:
        p = []
        for i in range(A.height):
            row = []
            for j in range(B.width):
                row = row + [polySum(A.ele[i][j], B.ele[i][j])]
            p = p + [row]

    C = matS(p)

    return C


'''
p=[[[1,1],[2],[3,2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[5,-1,-3]]]
A=matS(p)
p=[[1,2,6],[2,-1,3],[4,2,-5]]
B0=matN(p)
B=matN2S(B0)
A.print()
B.print()
C=matSSum(A,B)
C.print()
'''


def matSScale(f, A):
    # f(s): polynimail, A:matS
    # return: f(s)*A
    p = []
    for i in range(A.height):
        row = []
        for j in range(A.width):
            row = row + [polyTimes(f, A.ele[i][j])]
        p = p + [row]

    C = matS(p)

    return C


'''
p=[[[1,1],[2],[3,2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[5,-1,-3]]]
A=matS(p)
f=[1,2,6]
A.print()
print(f)
B=matSScale(f,A)
B.print()
'''


def matSTimes(A, B):
    # A,B: matS
    # return C=A*B
    if A.width != B.height:
        print('wrong sizes')
        5 / 0.0
    else:
        C = matSzero(A.height, B.width)
        for i in range(C.height):
            for j in range(C.width):
                Q = []
                for k in range(A.width):
                    Q = polyTimes(A.ele[i][k], B.ele[k][j])
                    C.ele[i][j] = polySum(C.ele[i][j], Q)
        return (C)


'''
p=[[[1,1],[2],[3,2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[5,-1,-3]],[[1],[2],[1,2,3]]]
A=matS(p)
A.print()
#p=[[[1,0],[2],[2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[-1,-3]]]
#B=matS(p)
B=matSzero(3,5)
#for j in range(5):
#    B.ele[0][j]=[1]
for i in range(3):
    B.ele[i][1]=[1]
B.print()
C=matSTimes(A,B)
C.print()
'''



# use a list to represent a polynomial


# sI-A is matSSum(matSScale([1,0],matSeye(A.height)),matSScale([-1],matN2S(A)))

def matNminor(A, i0, j0):
    # return the minor of A, say B
    # minor of A: remove the i0-row and j0-th column from A
    if A.height < 2:
        print('Size of A is not enouph to have a minor')
        4 / 0
    else:
        c = []
        for i in range(A.height):
            if i != i0:
                row = []
                for j in range(A.width):
                    if j != j0:
                        row = row + [A.ele[i][j]]
                c = c + [row]
        C = matN(c)
        # C.print()
        return C


'''
p=[[1,0,4,2],[2,12,31,3],[1,19,2,3],[2,1,3,-2]]
A=matN(p)
A.print()
B=matNminor(A,1,1)
B.print()
'''


def matNdet(A):
    # A: matN
    # return det(A), a number
    if A.height == 1:
        det = A.ele[0][0]
        # print(det)
    else:
        det = 0
        coe = 1
        for i in range(A.height):
            w0 = matNdet(matNminor(A, i, 0))
            w1 = A.ele[i][0] * w0
            w2 = coe * w1
            # print(w2)
            det = det + w2
            coe = -coe

    return det


'''
p=[[10,2,0],[2,3,0],[0,0,2]]
#p=[[3]]
A=matN(p)
A.print()
print(matNdet(A))
'''


def matSminor(A, i0, j0):
    # return the minor of A, say B
    # minor of A: remove the i0-row and j0-th column from A
    if A.height < 2:
        print('Size of A is not enouph to have a minor')
        4 / 0
    else:
        c = []
        for i in range(A.height):
            if i != i0:
                row = []
                for j in range(A.width):
                    if j != j0:
                        row = row + [A.ele[i][j]]
                c = c + [row]
        C = matS(c)
        # C.print()
        return C


'''
p=[[[1,1],[2],[3,2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[5,-1,-3]],[[1],[2],[1,2,3]]]
A=matS(p)
A.print()
B=matSminor(A,1,1)
B.print()
'''


def matSdet(A):
    # A: matS
    # return det(A), a polynomial of s
    if A.height == 1:
        det = A.ele[0][0]
        # print(det)
    else:
        det = [0]
        coe = 1
        for i in range(A.height):
            w0 = matSdet(matSminor(A, i, 0))
            w1 = polyTimes(A.ele[i][0], w0)
            w2 = polyScale(coe, w1)
            # print(w2)
            det = polySum(det, w2)
            coe = -coe

    return det


'''
p=[[10,2,0],[2,3,0],[0,0,2]]
#p=[[3]]
A=matN(p)
B=matN2S(A)
A.print()
B.print()
print(matSdet(B)[0])
print('---------------')

p=[[[1,5],[-4]],[[3],[1,0]]]
A=matS(p)
A.print()
print(matSdet(A))
'''

# ----------------------------------------------------------
# sI-A is matSSum(matSScale([1,0],matSeye(A.height)),matSScale([-1],matN2S(A)))

'''
n=4    #no more than 9, cost long time
print()
print(f'A has size {n}x{n}')
print()
A=matSzero(n,n)
for i in range(A.height):
    for j in range(A.width):
        A.ele[i][j]=randint(-9,9)
A.print()

I=matSeye(n)
sI=matSScale([1,0],I)
#sI.print()

nA=matSScale([-1],matN2S(A))
#nA.print()

sImA=matSSum(sI,nA)
#sImA.print()

chA=matSdet(sImA)
print(f'characteristic polynomial of A is {chA}')
'''


def matNcof(A):
    # A is a matN
    # return F, the matrix of cofactors of A
    if A.height != A.width:
        print('Not a square matrix')
        3 / 0
    else:
        if A.height == 1:
            p = [[1]]
            F = matN(p)
        else:
            F = matNzero(A.height, A.height)
            icoe = 1
            for i in range(F.height):
                coe = icoe
                for j in range(F.width):
                    Aqq = matNminor(A, j, i)
                    Aqd = matNdet(Aqq)
                    F.ele[i][j] = coe * Aqd
                    coe = -coe
                icoe = -icoe

    return F


'''
n=3    #no more than 8, cost long time
print()
print(f'A has size {n}x{n}')
print()
A=matNzero(n,n)
for i in range(A.height):
    for j in range(A.width):
        A.ele[i][j]=randint(-9,9)
A.print()

F=matNcof(A)
print('The matrix of cofactors of A is F = : ')
F.print()

C=matNTimes(A,F)
D=matNTimes(F,A)
print('A times F and F times A')
C.print()
D.print()

print('Determinant of A is:')
print(matNdet(A))
'''


def matNinv(A):
    # A is a matN
    # return inverse of A
    D = matNdet(A)
    F = matNcof(A)
    for i in range(F.height):
        for j in range(F.width):
            F.ele[i][j] = F.ele[i][j] / D
    return F


'''
n=3    #no more than 8, cost long time
print()
print(f'A has size {n}x{n}')
print()
A=matNzero(n,n)
for i in range(A.height):
    for j in range(A.width):
        A.ele[i][j]=randint(-9,9)
A.print()
B=matNinv(A)
print('inverse of A is B=:')
B.print()
C=matNTimes(A,B)
for i in range(C.height):
    for j in range(C.width):
        C.ele[i][j]=round(C.ele[i][j],4)
C.print()
'''


def matScof(A):
    # A is a matS
    # return F, the matrix of cofactors of A
    if A.height != A.width:
        print('Not a square matrix')
        3 / 0
    else:
        if A.height == 1:
            p = [[[1]]]
            F = matS(p)
        else:
            F = matSzero(A.height, A.height)
            icoe = 1
            for i in range(F.height):
                coe = icoe
                for j in range(F.width):
                    Aqq = matSminor(A, j, i)
                    Aqd = matSdet(Aqq)
                    F.ele[i][j] = polyScale(coe, Aqd)
                    coe = -coe
                icoe = -icoe

    return F


'''
n=3    #no more than 8, cost long time
print()
print(f'A has size {n}x{n}')
print()
A=matSzero(n,n)
for i in range(A.height):
    for j in range(A.width):
        A.ele[i][j]=[randint(0,1),randint(-9,9)]
A.print()

F=matScof(A)
print('The matrix of cofactors of A is F = : ')
F.print()

C=matSTimes(A,F)
D=matSTimes(F,A)
print('A times F and F times A')
C.print()
D.print()

print('Determinant of A is:')
print(matSdet(A))
'''


class SYStf:
    # represent a system by two polynomials
    def __init__(self, nn, dd):
        # nn and dd are polynomials, i.e., lists
        if len(nn) > 0:
            while nn[0] == 0 and len(nn) > 0:
                del nn[0]
        if len(dd) > 0:
            while dd[0] == 0 and len(dd) > 0:
                del dd[0]
        if len(nn) > len(dd):
            print('warning: numernator has higher order than denominator')
            2 / 0
        if dd[0] == 0:
            print('error: denominator is zero')
            2 / 0

        # denominator of the transfer function start from s^k, or: [1,?,...]
        nnn = []
        for i in range(len(nn)):
            nnn = nnn + [nn[i] / dd[0]]
        ddd = [1]
        for i in range(1, len(dd)):
            ddd = ddd + [dd[i] / dd[0]]

        self.num = nnn
        self.den = ddd

    def print(self):
        print(' ' * 22, self.num)
        print('Transfer function is ', '-' * (4 * len(self.den)))
        print(' ' * 22, self.den)


'''
a=[1,2]
b=[3,8,12]
G=SYStf(a,b)
print(a,b)
G.print()
'''


class SYSss:
    # represent a system by 4 matrices
    def __init__(self, A0, B0, C0, D0):
        # A0,B0,C0,D0 are matN

        self.A = matN(A0)
        self.B = matN(B0)
        self.C = matN(C0)
        self.D = matN(D0)

        if self.A.height != self.A.width:
            print('A would be a square matrix')
            2 / 0
        n = self.A.height
        if self.B.height != n:
            print('B would have the same height with A')
            2 / 0
        p = self.B.width
        if self.C.width != n:
            print('C would have the same width with A')
            2 / 0
        m = self.C.height
        if self.D.height != m or self.D.width != p:
            print('Wrong size of D')
            4 / 0

        self.inputDim = p
        self.outputDim = m
        # D as size mxp
        # we only consider SIMI case: p=m=1
        self.stateDim = n

    def print(self):
        self.A.print()
        self.B.print()
        self.C.print()
        self.D.print()
        print(f'Dim. of input is {self.inputDim}, ')
        print(f'Dim. of output is {self.outputDim}, ')
        print('Dim. of state is', self.stateDim, '.')

def plot(t,f):#todo make work better with more dynamic scaling
    M = 101
    dx = 0.1
    x = [i * dx for i in range(M)]
    y = [(3 - x[i]) ** 2 for i in range(M)]

    #t:times,
    #f(t), function
    import tkinter as fg
    win1 = fg.Tk()

    canv1 = fg.Canvas(width=600, height=500, bg='white')
    canv1.pack()

    #M=len(t)
    Wid=t[M-1]
    scalet=480/Wid
    scaley=10
    for i in range(M-1):
        canv1.create_line(10+scalet*t[i],250-scaley*y[i],10+scalet*t[i+1],250-scaley*y[i+1], fill='green')
    win1.mainloop()
    #todo, ^ will prevent other code from running?

def matNnorm(A):
    # A: matN
    # return max(abs(A[i][j])
    width = A.width
    height = A.height
    max = -999999999
    for i in range(width):
        for j in range(height):
            val = abs(A.ele[i][j])
            if val>max:
                max = val
    return max

def matNrand(width,height):
    #Returns a matN object of width and height with random values between 1 and 9 at each entry
    A = []
    for i in range(height):
        row = [random.random() for j in range(width)]
        A.append(row)
    return matN(A)

def matNcorr(A):
    #given some matrix A, corrects (rounds) all entries:
    Acopy = [None] * A.height
    for i in range(A.height):
        row = [None] * A.width
        for j in range(A.width):
            val = A.ele[i][j]
            if(abs(val) < 0.001):
                val=0.0
            else:
                val = round(val,5)
            row[j] = val
        Acopy[i]=row
    B = matN(Acopy)
    return(B)