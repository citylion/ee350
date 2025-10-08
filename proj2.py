from jinlib import *

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