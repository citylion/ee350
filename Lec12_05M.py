from Lec06_03M import *
from Lec06_03MBox import *
from random import randint

def roots(f):
    #f: polynomial
    #returen list of poles and their orders
    K=len(f)
    if K==2:
        rtA=[-f[1]/f[0]]
        odA=[1]
    elif K==3:
        a1=(f[1]**2-4*f[0]*f[2])**0.5
        p1=(-f[1]+a1)/(2*f[0])
        p2=(-f[1]-a1)/(2*f[0])
        rtA=[p1,p2]
        odA=[1,1]
    else:
        OK=0
        while OK==0:
            g=[1,randint(-50,50),randint(-50,50)]

            T=1
            M=3
            [u,v]=polyDivid(f,g)
            #we want v=0, then f=g*u, g is order 2 and u is order K-2
            er0=abs(v[0])+abs(v[1])
            kkk=1
            while er0>1e-14 and kkk<500:
                kkk=kkk+1
                for i in range(-M,M+1):
                    for j in range(-M,M+1):
                        g1=[1,g[1]+i*T,g[2]+j*T]
                        [u1,v1]=polyDivid(f,g1)
                        if len(v1)==1:
                            v1=[0]+v1
                        er1=abs(v1[0])+abs(v1[1])
                        if er1<=er0:
                            er0=er1
                            i0=i
                            j0=j
            
                if i0==0 and j0==0:
                    T=T/10
                else:
                    g=[1,g[1]+i0*T,g[2]+j0*T]

                #print(i0,j0,g,er0)

            #print(er0)
            if er0>1e-14:
                OK=0
            else:
                OK=100
        
        [u,v]=polyDivid(f,g)


        [rt1,od1]=roots(g)
        [rt2,od2]=roots(u)
        rtA=rt1+rt2
        odA=od1+od2

    return(rtA, odA)



print(roots([2,4]))
print(roots([2,4,0]))
print(roots([1,4,4]))
print(roots([1,4,4,0,0]))
print(roots([1,4,4,6,10]))
print(roots([1,0,0,4,0,0,4,0,0]))

    
        
        
    
