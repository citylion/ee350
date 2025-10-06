from random import randint
from Lec06_03MBox import *


#use a list to represent a polynomial


#sI-A is matSSum(matSScale([1,0],matSeye(A.height)),matSScale([-1],matN2S(A)))

def matNminor(A,i0,j0):
    #return the minor of A, say B
    #minor of A: remove the i0-row and j0-th column from A
    if A.height<2:
        print('Size of A is not enouph to have a minor')
        4/0
    else:
        c=[]
        for i in range(A.height):
            if i!=i0:
                row=[]
                for j in range(A.width):
                    if j!=j0:
                        row=row+[A.ele[i][j]]
                c=c+[row]
        C=matN(c)
        #C.print()
        return C


'''
p=[[1,0,4,2],[2,12,31,3],[1,19,2,3],[2,1,3,-2]]
A=matN(p)
A.print()
B=matNminor(A,1,1)
B.print()
'''

def matNdet(A):
    #A: matN
    #return det(A), a number
    if A.height==1:
        det=A.ele[0][0]
        #print(det)
    else:
        det=0
        coe=1
        for i in range(A.height):
            w0=matNdet(matNminor(A,i,0))
            w1=A.ele[i][0]*w0
            w2=coe*w1
            #print(w2)
            det=det+w2
            coe=-coe
    
    return det


'''
p=[[10,2,0],[2,3,0],[0,0,2]]
#p=[[3]]
A=matN(p)
A.print()
print(matNdet(A))
'''


def matSminor(A,i0,j0):
    #return the minor of A, say B
    #minor of A: remove the i0-row and j0-th column from A
    if A.height<2:
        print('Size of A is not enouph to have a minor')
        4/0
    else:
        c=[]
        for i in range(A.height):
            if i!=i0:
                row=[]
                for j in range(A.width):
                    if j!=j0:
                        row=row+[A.ele[i][j]]
                c=c+[row]
        C=matS(c)
        #C.print()
        return C

'''
p=[[[1,1],[2],[3,2,1]],[[1,2],[3,1],[3,-2]],[[4],[4],[5,-1,-3]],[[1],[2],[1,2,3]]]
A=matS(p)
A.print()
B=matSminor(A,1,1)
B.print()
'''
            

def matSdet(A):
    #A: matS
    #return det(A), a polynomial of s
    if A.height==1:
        det=A.ele[0][0]
        #print(det)
    else:
        det=[0]
        coe=1
        for i in range(A.height):
            w0=matSdet(matSminor(A,i,0))
            w1=polyTimes(A.ele[i][0],w0)
            w2=polyScale(coe,w1)
            #print(w2)
            det=polySum(det,w2)
            coe=-coe
    
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

#----------------------------------------------------------
#sI-A is matSSum(matSScale([1,0],matSeye(A.height)),matSScale([-1],matN2S(A)))

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
    #A is a matN
    #return F, the matrix of cofactors of A
    if A.height!=A.width:
        print('Not a square matrix')
        3/0
    else:
        if A.height==1:
            p=[[1]]
            F=matN(p)
        else:
            F=matNzero(A.height,A.height)
            icoe=1
            for i in range(F.height):
                coe=icoe
                for j in range(F.width):
                    Aqq=matNminor(A,j,i)
                    Aqd=matNdet(Aqq)
                    F.ele[i][j]=coe*Aqd
                    coe=-coe
                icoe=-icoe
                    
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
    #A is a matN
    #return inverse of A
    D=matNdet(A)
    F=matNcof(A)
    for i in range(F.height):
        for j in range(F.width):
            F.ele[i][j]=F.ele[i][j]/D
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
    #A is a matS
    #return F, the matrix of cofactors of A
    if A.height!=A.width:
        print('Not a square matrix')
        3/0
    else:
        if A.height==1:
            p=[[[1]]]
            F=matS(p)
        else:
            F=matSzero(A.height,A.height)
            icoe=1
            for i in range(F.height):
                coe=icoe
                for j in range(F.width):
                    Aqq=matSminor(A,j,i)
                    Aqd=matSdet(Aqq)
                    F.ele[i][j]=polyScale(coe,Aqd)
                    coe=-coe
                icoe=-icoe
                    
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
    #represent a system by two polynomials
    def __init__(self, nn, dd):
        #nn and dd are polynomials, i.e., lists
        if len(nn)>0:
            while nn[0]==0 and len(nn)>0:
                del nn[0]
        if len(dd)>0:
            while dd[0]==0 and len(dd)>0:
                del dd[0]
        if len(nn)>len(dd):
            print('warning: numernator has higher order than denominator')
            2/0
        if dd[0]==0:
            print('error: denominator is zero')
            2/0
        
        #denominator of the transfer function start from s^k, or: [1,?,...]
        nnn=[]
        for i in range(len(nn)):
            nnn=nnn+[nn[i]/dd[0]]
        ddd=[1]
        for i in range(1,len(dd)):
            ddd = ddd+[dd[i]/dd[0]]
            
        self.num = nnn
        self.den = ddd
        
    def print(self):
        print(' '*22,self.num)
        print('Transfer function is ','-'*(4*len(self.den)))
        print(' '*22,self.den)
        

'''
a=[1,2]
b=[3,8,12]
G=SYStf(a,b)
print(a,b)
G.print()
'''



class SYSss:
    #represent a system by 4 matrices
    def __init__(self, A0, B0, C0, D0):
        #A0,B0,C0,D0 are matN

            
        self.A = matN(A0)
        self.B = matN(B0)
        self.C = matN(C0)
        self.D = matN(D0)

        if self.A.height!=self.A.width:
            print('A would be a square matrix')
            2/0
        n=self.A.height
        if self.B.height!=n:
            print('B would have the same height with A')
            2/0
        p=self.B.width
        if self.C.width!=n:
            print('C would have the same width with A')
            2/0
        m=self.C.height
        if self.D.height!=m or self.D.width!=p:
            print('Wrong size of D')
            4/0

        self.inputDim = p
        self.outputDim = m
        #D as size mxp
        #we only consider SIMI case: p=m=1
        self.stateDim=n

    def print(self):
        self.A.print()
        self.B.print()
        self.C.print()
        self.D.print()
        print(f'Dim. of input is {self.inputDim}, ')
        print(f'Dim. of output is {self.outputDim}, ')
        print('Dim. of state is',self.stateDim,'.')
            

'''
A=[[1,2,3],[2,3,4],[1,2,1]]
B=[[1],[0],[0]]
C=[[0,1,2]]
D=[[3]]
G=SYSss(A,B,C,D)
G.print()
'''


def ss2tf(G):
    #G is SYSss
    #transfer function is C(SI-A)^(-1)B+D
    #return a SYStf
    As=matN2S(G.A)
    Bs=matN2S(G.B)
    Cs=matN2S(G.C)
    Ds=matN2S(G.D)

    #SI-A
    I=matSeye(G.stateDim) 
    sI=matSScale([1,0],I)
    nA=matSScale([-1],As)
    sImA=matSSum(sI,nA)
    #sImA.print()

    #det(SI-A), denominator of the transfer function
    den=matSdet(sImA)
    #print(den) #it is a polynomial

    #numerator of tf==>C(SI-A)^(-1)B+D
    Fs=matScof(sImA)
    #WW=matSTimes(Fs,sImA)
    #WW.print()
    CF=matSTimes(Cs,Fs)
    CFB=matSTimes(CF,Bs)
    chD=matSScale(den,Ds)
    Ns=matSSum(CFB,chD) #it is a matS, same size of D

    Gtf=SYStf(Ns.ele[0][0],den)
    #This is actually the transfer function from the first input to the first output
    #for MIMO case, m>1 or p>1, and we can have
    #Gtf[i][j]=SYStf(Ns.ele[i][j],den), i:0 to m-1, j: 0 to p-1
    return Gtf

#A=[[0,1,0],[0,0,1],[-8,-2,-6]]
A=[[6,4],[-2,0]]
B=[[7],[-9]]
C=[[1,0]]
D=[[2]]
G=SYSss(A,B,C,D)
G.print()
G1=ss2tf(G)
G1.print()

#input('press enter key to quit  ')

