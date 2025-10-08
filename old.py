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


'''
A=[[1,2,3],[2,3,4],[1,2,1]]
B=[[1],[0],[0]]
C=[[0,1,2]]
D=[[3]]
G=SYSss(A,B,C,D)
G.print()
'''

#from gabe
def tf2ss(tf):
    """
    Convert transfer function (SYStf) to state-space (SYSss).
    Uses controllable canonical form.
    """
    den = tf.den   # Extract denominator coefficients
    num = tf.num   # Extract numerator coefficients
    n = len(den)-1   # System order (degree of denominator)

    if den[0] != 1:
        # normalize if leading coeff not 1
        num = [x/den[0] for x in num]
        den = [x/den[0] for x in den]

    # A matrix
    A = [[0]*n for _ in range(n)]   # Initialize n x n zeros
    for i in range(n-1):
        A[i][i+1] = 1   # Set 1s on superdiagonal
    A[-1] = [-c for c in list(reversed(den[1:]))]    # Last row = negated denominator coeffs


    # B vector
    B = [[0] for _ in range(n)]   # Column vector of zeros
    B[-1][0] = 1   # Last element = 1

    # Adjust numerator length
    num = [0]*(n+1-len(num)) + num
    # D scalar
    D = [[num[0]]]   # First numerator term = D
    # C vector
    C = [ [num[k] - den[k]*num[0] for k in range(n)] ]
    # Return new state-space system
    return SYSss(A,B,C,D)

def ss2tf(G):
    # G is SYSss
    # transfer function is C(SI-A)^(-1)B+D
    # return a SYStf
    As = matN2S(G.A)
    Bs = matN2S(G.B)
    Cs = matN2S(G.C)
    Ds = matN2S(G.D)

    # SI-A
    I = matSeye(G.stateDim)
    sI = matSScale([1, 0], I)
    nA = matSScale([-1], As)
    sImA = matSSum(sI, nA)
    # sImA.print()

    # det(SI-A), denominator of the transfer function
    den = matSdet(sImA)
    # print(den) #it is a polynomial

    # numerator of tf==>C(SI-A)^(-1)B+D
    Fs = matScof(sImA)
    # WW=matSTimes(Fs,sImA)
    # WW.print()
    CF = matSTimes(Cs, Fs)
    CFB = matSTimes(CF, Bs)
    chD = matSScale(den, Ds)
    Ns = matSSum(CFB, chD)  # it is a matS, same size of D

    Gtf = SYStf(Ns.ele[0][0], den)
    # This is actually the transfer function from the first input to the first output
    # for MIMO case, m>1 or p>1, and we can have
    # Gtf[i][j]=SYStf(Ns.ele[i][j],den), i:0 to m-1, j: 0 to p-1
    return Gtf


# A=[[0,1,0],[0,0,1],[-8,-2,-6]]
'''
A = [[6, 4], [-2, 0]]
B = [[7], [-9]]
C = [[1, 0]]
D = [[2]]
G = SYSss(A, B, C, D)
G.print()
G1 = ss2tf(G)
G1.print()
'''

# input('press enter key to quit  ')

def roots(f):
    # f: polynomial
    # return list of poles and their orders
    K = len(f)
    if K == 2:
        rtA = [-f[1] / f[0]]
        odA = [1]
    elif K == 3:
        a1 = (f[1] ** 2 - 4 * f[0] * f[2]) ** 0.5
        p1 = (-f[1] + a1) / (2 * f[0])
        p2 = (-f[1] - a1) / (2 * f[0])
        rtA = [p1, p2]
        odA = [1, 1]
    else:
        OK = 0
        while OK == 0:
            g = [1, randint(-50, 50), randint(-50, 50)]

            T = 1
            M = 3
            [u, v] = polyDivid(f, g)
            # we want v=0, then f=g*u, g is order 2 and u is order K-2
            er0 = abs(v[0]) + abs(v[1])
            kkk = 1
            while er0 > 1e-14 and kkk < 500:
                kkk = kkk + 1
                for i in range(-M, M + 1):
                    for j in range(-M, M + 1):
                        g1 = [1, g[1] + i * T, g[2] + j * T]
                        [u1, v1] = polyDivid(f, g1)
                        if len(v1) == 1:
                            v1 = [0] + v1
                        er1 = abs(v1[0]) + abs(v1[1])
                        if er1 <= er0:
                            er0 = er1
                            i0 = i
                            j0 = j

                if i0 == 0 and j0 == 0:
                    T = T / 10
                else:
                    g = [1, g[1] + i0 * T, g[2] + j0 * T]

                # print(i0,j0,g,er0)

            # print(er0)
            if er0 > 1e-14:
                OK = 0
            else:
                OK = 100

        [u, v] = polyDivid(f, g)

        [rt1, od1] = roots(g)
        [rt2, od2] = roots(u)
        rtA = rt1 + rt2
        odA = od1 + od2

    return (rtA, odA)

def evalPoly(f,x):
    # given a list of numbers representing a polynomial,
    # evalulate the polynomial's value for input x
    # return number = f(x)
    powr = len(f)-1
    result = 0
    for coef in f:
        if(powr == 0):
            result = result + coef
        else:
            result = result + (coef * (pow(x, powr)))

        powr = powr-1

    assert powr == -1 #at end of process, powr should be 0 (last term)
    return result

def tf2yt(G):
    #G a tf object

    num = G.num
    den = G.den
    root = roots(den)
    maxOrder = 1
    for order in root[1]:
        if order > maxOrder:
            maxOrder=order
    if maxOrder == 1:
        # make new list, for each root, find its numerator PFD
        cof = [0] * len(root[0]) #pfd coefficients corresponding to each root
        pos = 0
        for rt in root[0]:#for each root, find pfd
            pnum = 1 #pfd calculation numerator
            pden = 1 #pfd calculation denominator
            for rta in root[0]:#for each other root
                if rt == rta:
                    continue
                pden = pden*(rt -rta)
            #finished calculating denominator
            pnum = evalPoly(num,rt)
            cof[pos] = (pnum/pden)
            pos=pos+1
        prt = "" #to print
        pos = 0
        firstPrintedTerm = True
        for rt in root[0]:
            if not cof[pos] == 0: #if the exponential's coefficient is 0, skip printing
                if not firstPrintedTerm: #separate each term of the output with +
                    prt = prt + " + "
                prt = prt + str(round(cof[pos],5)) + "*" + "exp(" + str(round(rt,5)) + "*t" +") "
                firstPrintedTerm = False
            pos = pos + 1
        print(prt)
    else:
        print('Error, ss2yt currently supports roots only of form (s+k)^1 and not (s+k)^2 or higher')


def tfStepResponse(tf):
    tf1 = tf2tfOneOverS(tf)
    tf2yt(tf1)

def tf2tfOneOverS(G):
    # may have a situation where we need to multiply tf by 1/s
    # however, we cannot represent 1/s with our list [] as the last term supposed is just s^0 and never s^-1
    # but, the equivalent can be done by multiply the numerator by s and denominator by s^2
    # or to be more specific to the implementation needed, shifting the numerator left by 1, and the denominator left by 2.
    # input SYStf
    # output SYStf
    return SYStf(polyShift(G.num,1),polyShift(G.den,2))

def polyShift(A, offset):
    # returns a shifted version of list A, shifted left by offset
    arrLength = len(A)
    B = [0] * (arrLength + offset)
    for i in range(arrLength):
        B[i] = A[i]
    return B

def tfPrint(E):
    xlx = max(len(E.num), len(E.den))
    print(E.num)
    print('-' * xlx * 5)
    print(E.den)

def normal2tf(lhs,rhs):
    #lhs, rhs, list representing polynomial
    return SYStf(rhs,lhs)

'''
A = [2, 0, -16]
B = roots(A)
print(B)
'''

'''
A = [[7,2],[5,4]]
B = [[0],[3]]
C = [[7,0]]
D = [[0]]

E = ss2tf(SYSss(A,B,C,D))
tf2yt(E)
'''

'''
xlx = max(len(E.num),len(E.den))
print(E.num)
print('-'*xlx*5)
print(E.den)
'''
'''
#PROJECT / QUIZ 2 
#Sample for solving Quiz Question 1:
print(" ")
lhs = [1, 8, 5] #the left hand side of the equation (coefficients of y(t) given by quiz problem 1)
rhs = [6, 8] #the right hand side of the equation (coefficients of r(t) given by quiz problem 2)
tf1 = normal2tf(lhs,rhs)
tf2yt(tf1)
# ^ prints resulting expression
print(" ")
print(" ")

#SAmple for solving  Quiz Question 2:
lhs = [1, 6.3, 2] #the left hand side of the equation (coefficients of y(t) given by quiz problem 1)
rhs = [5, 3] #the right hand side of the equation (coefficients of r(t) given by quiz problem 2)
tf2 = normal2tf(lhs,rhs)
tfStepResponse(tf2)
# ^ prints resulting expression
print(" ")
print(" ")
#Sample for solving Quiz Question 3:
A = [[4,7],[4,5]]
B = [[0],[3]]
C = [[4,0]]
D = [[0]]
ss3 = SYSss(A,B,C,D)
tf3 = ss2tf(ss3)
tf2yt(tf3)
# ^ prints resulting expression

print(" ")
print(" ")


#Sample for solving Quiz Question 4:
A = [[5,8],[5,5]]
B = [[0],[7]]
C = [[3,0]]
D = [[0]]
ss4 = SYSss(A,B,C,D)
tf4 = ss2tf(ss4)
tfStepResponse(tf4)
# ^ prints resulting expression
'''

###### Project 3
#################

def matSinv(A):
    # A is a matS
    # return inverse of A
    D = matSdet(A)
    F = matScof(A)
    for i in range(F.height):
        for j in range(F.width):
            F.ele[i][j] = F.ele[i][j] / D
    return F

#given some array of arrays [[], [], []], representing a coefficient of an exp
#evaluates it in the form Aexp^pt + Bexp^qt summing all terms then returning the result
#for some given t
def evalExpSum(A,t):
    sm = 0
    for entry in A:
        sm = sm + A[0]*math.exp(A[1])*t
    return sm


#why make a second function?
#this one returns a list of lists [[], [], []]
#where the inner array contains two values, [[a,b]
#representing Ae^b
#instead of printing to console like tf2yt, this one actually returns the data
def tf2yt2(G):
    #G a tf object

    num = G.num
    den = G.den
    root = roots(den)
    maxOrder = 1
    for order in root[1]:
        if order > maxOrder:
            maxOrder=order
    if maxOrder == 1:
        # make new list, for each root, find its numerator PFD
        cof = [0] * len(root[0]) #pfd coefficients corresponding to each root
        pos = 0
        for rt in root[0]:#for each root, find pfd
            pnum = 1 #pfd calculation numerator
            pden = 1 #pfd calculation denominator
            for rta in root[0]:#for each other root
                if rt == rta:
                    continue
                pden = pden*(rt -rta)
            #finished calculating denominator
            pnum = evalPoly(num,rt)
            cof[pos] = (pnum/pden)
            pos=pos+1


        count=0
        for rt in root[0]:
            if not cof[pos] == 0: #if the exponential's coefficient is 0, skip printing
              count=count+1
        D = [None]*count #create an empty list []
        for rt in root[0]:
            if not cof[pos] == 0:  # if the exponential's coefficient is 0, skip printing
                D = [round(cof[pos],5),round(rt,5)]
        return D
    else:
        print('Error, ss2yt2 currently supports roots only of form (s+k)^1 and not (s+k)^2 or higher')

def sIminusA(A: matS):
    # A a matN object, some matrix that should be square
    # Returns a new matrix s*I - A

    B = [None]*A.width
    for i in range(A.width):
        B[i] = [None]*A.height
        for j in range(A.height):
            if i == j: #special case
                B[i][j] = [1, -A.ele[i][j]]
            else:
                B[i][j] = [-A.ele[i][j]]

    return matS(B)


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


#The difference between functions PhiT, and PhiTn is that
#PhiT picks a value of n, say n=300.
#higher n results in a more accurate result.
def PhiT(A,t):
    return PhiTn(A,t,300)

def PhiTn(A, t, n):
    debug=False
    # A: matN, t:number, n steps total
    # return I+At+A^2t^2/2+A^3t^3/6+A^4t^4/4!+.....

    At = matNScale(t, A)
    N = A.height
    I = matNeye(N)
    Phi = matNSum(I, At)
    i=2;
    fact=1;
    if n < 2:
        print("PhiT, n should be at least 2")
        n=2
    while i<=n:
        #say sub-expression is
        #1/(2!) A^2 t^2 (for i==2)
        #fact is factorial 1/(x!), Amult is A^2, tmult is t^2
        fact = fact*i
        try:
            factinv = 1.0 / math.factorial(i)
        except OverflowError:
            return Phi
        if factinv == 0.0:
            break
        #now A^i
        Amult = matN(A.ele)
        for j in range(i-1):#pretty sure this -1 should be here
            Amult = matNTimes(Amult,A)
        tmult = t**i
        ith_term = matNScale(tmult*factinv,Amult)

        phiclone = matN(Phi.ele)
        Phi = matNSum(Phi,ith_term)

        if debug or i == n:
            print("PhiT step " + str(i))
            print("t: " + str(t))
            print("i: " + str(i))
            print("factinv: " + str(factinv))
            print("tmult: " + str(tmult))

            print("--->")
            Phi.print()
        i=i+1

    Phi = matNcorr(Phi)
    return (Phi)

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


A=matNScale(4,matNrand(4,4))
#A= matNeye(4)
A.print()
t1=-1.3
B1=PhiT(A,t1)
print("Matrix B1:")
B1.print()

t2=1.3
B2=PhiT(A,t2)
print("Matrix B2:")
B2.print()


C=PhiT(A,t1+t2)
print("Matrix C:")
C.print()
D= matNTimes(B1,B2)
D = matNcorr(D)
print("Matrix D:")
D.print()
