import math

from jinlib import *
from proj2 import *

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
    return PhiTn(A,t,2000)

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

def SYSout_num(G,t,u,y0):
    #G:system
    #t:list of time
    #u:input signal for time t
    #len(u)=len(t)
    #y0:list,initial, len(y0)=G.order
    #For tf system, y0 is y(t=0). For ss system,  y0 is x at t=0
    #if G is tf type, G1=tf2ss(G), and x0 = Ob\y0
    #if G is ss type, G1=G, x0=y0



    if isinstance(G,SYSss):
        dt = t[1]-t[0]
        t_final = t[len(t)-1]
        steps = t_final/dt + 1
        steps = math.floor(steps) #steps should be an integer


        x0 = [[0]] * len(y0)
        for i in range(len(y0)):
            x0[i][0] = y0[i]


        return SYSssout_num(G,dt,steps,u,x0)
        #todo: properly handle case where t_n isnt dt*n (increase in t isnt uniform)

    elif isinstance(G,SYStf):
        print("not yet implemented")

    #loop:
    #   x[n+1]=Phi*x[n]+Beta*u[n]
    #   y[n]=C*x[n]+D*u[n]

    y=[a for a in u]

    return(y)

def SYSssout_num(G,dt,n,u,x0):
    # G:SYSss
    # dt: time step
    # n: #steps t_final = n*dt
    # u:input signal for time t
    # len(u)=len(t)
    # x0 state vector given

    G1 = G # <- idk
    A=G1.A
    B=G1.B
    C=G1.C
    D=G1.D

    Phi=PhiT(A,dt)
    Beta = matNScale(dt,matNTimes(Phi,B))
    #Beta = matNScale(dt,B)

    x = [None]*n
    y = [None]*n

    x[0] = matN(x0)
    y[0] = matNSum(matNTimes(C,x[0]),matNTimes(D,matN([[u[0]]])))

    #loop starts at i=1
    for i in range(1,n):

        x[i] = matNSum(matNTimes(Phi, x[i - 1]), matNTimes(Beta, matN([[u[i]]])))
        y[i] = matNSum(matNTimes(C, x[i]), matNTimes(D, matN([[u[i]]])))

    return y

def uOfT(k,T,N,pwr):
    dt = T / (N - 1)
    return [k * ((i * dt) ** pwr) for i in range(N)]
'''
A=matNScale(4,matNrand(4,4))
#A= matNeye(4)
A.print()
t1=-1.3
B1=PhiT(A,t1)

t2=1.3
B2=PhiT(A,t2)

C=PhiT(A,t1+t2)
print("Matrix C:")
C.print()
D= matNTimes(B1,B2)
D = matNcorr(D)
print("Matrix D:")
D.print()
'''

'''
a=randint(-3,3)
b=randint(-3,3)
c=randint(-3,3)
d=randint(-3,3)
u=randint(-3,3)
v=randint(-3,3)
g=randint(-3,3)
h=randint(-3,3)

A=[[a,b],[c,d]]
B=[[u],[v]]
C=[[g,h]]
D=[[randint(-3,3)]]

G=SYSss(A,B,C,D)

T=randint(10,15)
M=20000
dt=T/(M-1)
t=[i*dt for i in range(M)]
u=[math.sin(3*t[i]) for i in range(M)]
y0=[randint(-3,3) for i in range(2)]
y=SYSout_num(G,t,u,y0)
print(f'when t={t[-1]},output y(t) = {y[-1]}')

plot(t,y)
'''

#problem 1
tf1 = normal2tf([2,5,21],[1,5,0])
ss1 = tf2ss(tf1)

T=15
M=1000
dt=T/(M-1)
t=[i*dt for i in range(M)]
u=uOfT(122,T,M,1)

out1 = SYSout_num(ss1,t,u,[2, 0])
val = out1[len(out1)-1]
print(val.ele)


#problem 2
tf2 = normal2tf([1,9,29,66],[1,9,0,0])
ss2 = tf2ss(tf2)

T=15
M=10000
dt=T/(M-1)
t=[i*dt for i in range(M)]
u=uOfT(121,T,M,2)
y0 = [0,1,0]

out2 = SYSout_num(ss2,t,u,y0)
val2 = out2[len(out2)-1]
print(val2.ele)

#problem 3
A = [[0,1],[-20,-9]]
B=[[0],[1]]
C=[[-20,0]]
D=[[1]]
ss3 = SYSss(A,B,C,D)

T=12
y0=[-2,2]
u=uOfT(135,T,M,1)
out3 = SYSout_num(ss3,t,u,y0)
val3 = out3[len(out3)-1]
print(val3.ele)


#problem 4
A = [[0,1,0],[0,0,1],[-56,-24,-7]]
B=[[0],[0],[1]]
C=[[-56,-24,0]]
D=[[0]]
ss4 = SYSss(A,B,C,D)

T=19
y0=[1,2,0]
u=uOfT(168,T,M,2)
out4 = SYSout_num(ss4,t,u,y0)
val4 = out4[len(out4)-1]
print(val4.ele)