# 2D TRUSS ANALYSIS BY USING STIFFNESS METHOD
# BATUHAN ÖZKAN
import numpy as np
import numpy.linalg as lin

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import pylab as pl

n=7 #Joint numbers
m=11# Member numbers


#Determination of Cross section area.

# Assuming section as the tube
# Properties are identical for 11 members.
D=0.05 # m
A=((np.pi)*(D**2))/4  #Cross Section area (m2)

E=2*10**8 # kN/m2
XY_values=np.array([0.0,0.0, 1.0,2.0, 2.0,0.0, 3.0,2.0, 4.0,0.0, 5.0,2.0, 6.0,0.0])
NC_values=np.array([1 ,2 ,1 ,3 ,2 ,3, 2, 4, 3, 4, 3, 5, 4, 5, 4, 6, 5, 6, 5, 7, 6, 7])

MG_values=np.array([E ,A ,E ,A ,E ,A, E, A, E, A, E, A, E, A, E, A, E, A, E, A, E, A])

BC_values=np.array([1, 1, 1, 7, 1, 1])
#First column represents joint number(I would want to starting number with 0.
#Second column represents is at x direction prescribed(1) or free(0)?
#Third column represents is at y direction prescribed(1) or free(0)?

FEXT_values=np.array([2,0.0,-11.0, 4,0.0,-22.0, 6,0,-11.0])

# calculating displacement matrix
def calculate_Df(Kff,Fext,Kfp,Dp):

    division_matrix = np.subtract(Fext,(Kfp.dot(Dp)))
    matrix = lin.solve(Kff,division_matrix)
    print("Kfp=")
    print(Kfp)
    return matrix

#calculating reaction matrix
def calculate_R(Kpf,Df,Kpp,Dp):
    matrix = np.dot(Kpf,Df) + np.dot(Kpp,Dp)
    return matrix

#calculating member end forces
def calculate_memberEndForce_q(k_prime,T,U):
    matrix = np.dot(k_prime,np.dot(T,U))

    return matrix


def assemble_matrix_1_to_2_using_Ce(Ce,k,K):
    size_array_k=k.shape
    row_k=size_array_k[0]
    col_k=size_array_k[1]

    # loop over k matrix's elements
    for i in range(0,row_k,1):
        for j in range(0,col_k,1):
            # defining indices of K value by using DOF_numbers
            K_index_1 = int(Ce[j]-1.0)
            K_index_2 = int(Ce[i]-1.0)

            K_value = K[K_index_1,K_index_2]

            # adding previous K value to new one recursively
            K[K_index_1][K_index_2] = K_value + k[i][j]

    return K


# defining el_stiff matrix at global coordinate to create element stiffness matrix
def el_stiff(T,k_prime):
    T_transpose = np.transpose(T)
    matrix = np.dot(T_transpose,np.dot(k_prime,T))

    return matrix

# defining el_stiff_local function to create element stiffness matrix
def el_stiff_local(A,E,x,y):
    L=findLength(x,y)
    ael=(A*E)/L
    matrix=[[ael,(-1)*ael],[(-1)*ael,ael]]
    return matrix

# defining T_matrix function to create transformation matrix
def T_matrix(x,y):

    delta = find_delta_Values(x,y)
    delta_x = delta[0]
    delta_y = delta[1]

    L = (findLength(x,y))

    cos_theta = delta_x / L
    sin_theta = delta_y / L

    matrix = [[cos_theta,sin_theta,0,0],[0,0,cos_theta,sin_theta]]

    return matrix

# defining findLength function to find length
def findLength(x,y):
    delta=find_delta_Values(x,y)
    delta_x=delta[0]
    delta_y=delta[1]
    L_value=np.sqrt(((delta_x)**2) + ((delta_y)**2))

    return L_value

#defining find_delta_Values function to find delta_x and delta_y
def find_delta_Values(x,y):
    start_x = x[0]
    end_x = x[1]
    start_y = y[0]
    end_y = y[1]
    delta_x = end_x[0] - start_x[0]
    delta_y = end_y[0] - start_y[0]
    delta = [delta_x , delta_y]

    return delta

# defining createMatrix function to create matrices by using row number,column number and values
def createMatrix(row,column,values):

    matrix=np.zeros([row, column], dtype =float)
    counter = 0                      # counter for element number in values
    for i in range(0,row,1):
        for j in range(0,column,1):
            matrix[i][j] = values[counter]
            counter = counter + 1
    return matrix

# creating matrices by using createMatrix function with row number,column number and matrix values
XY=createMatrix(n,2, XY_values)
NC=createMatrix(m,2, NC_values)
MG=createMatrix(m,2, MG_values)
BC=createMatrix(2,3, BC_values)
FEXT=createMatrix(3,3, FEXT_values)

#print(FEXT)
size_array_BC=BC.shape
row_BC=size_array_BC[0]
col_BC=size_array_BC[1]


DOF=np.zeros((n,2),dtype=float)

k=1
displacement_counter = 0
for i in range(0,n,1):     #loop over nodes
    BC_state= "not_exist"
    for j in range(0,row_BC,1): #loop over rows of BC array
        if BC[j,0]==i+1:
            BC_state="exist"
            x_value=j

    if BC_state=="not_exist": # If there is no BC on the node
        DOF[i,0]=k
        DOF[i,1]=k+1
        k=k+2
        displacement_counter = displacement_counter +2

    elif BC_state=="exist" and BC[x_value,1]==0: # else if there is BC on the node and X direction is not constrained.
        DOF[i,0]=k
        k=k+1
        displacement_counter = displacement_counter +1
    elif BC_state=="exist" and BC[x_value,2]==0: # else if there is BC on the node and Y direction is not constrained.
        DOF[i,1]=k
        k=k+1
        displacement_counter = displacement_counter +1
    #filling the remaining 0 entries of the DOF matrix
print(DOF)
for i in range(0,n,1):    #loop over nodes
    for j in range(0,2,1): # loop over X and Y directions
        if DOF[i,j]==0:
            DOF[i,j]=k
            k=k+1


size_array_DOF=DOF.shape
row_DOF=size_array_DOF[0]
col_DOF=size_array_DOF[1]

q=row_DOF*col_DOF

size_array_NC=NC.shape
row_NC=size_array_NC[0]
col_NC=size_array_NC[1]

K=np.zeros((q,q),dtype=float)

T_trans=[]
k_prime_add=[]
# Construction of K
for e in range(0,row_NC,1):         # loop over elements
    # defining required variables
    E=MG[e,0]
    A=MG[e,1]


    bn_id=int(NC[e,0]-1.0)
    en_id=int(NC[e,1]-1.0)

    x=np.array([[XY[bn_id,0]],[XY[en_id,0]]])
    y=np.array([[XY[bn_id,1]],[XY[en_id,1]]])


    k_prime=el_stiff_local(A,E,x,y)
    k_prime_add.append((k_prime))
    T=T_matrix(x,y)
    k=el_stiff(T,k_prime)
    #k_mult.append(k)
    T_trans.append(T)
    Ce=[DOF[bn_id,0],DOF[bn_id,1],DOF[en_id,0],DOF[en_id,1]]
    K=assemble_matrix_1_to_2_using_Ce(Ce,k,K)
    #print(Ce)
    #print(K)
#print(np.linalg.det(K))
#print("The System Stiffness Matrix K =")
#print(K,"N/mm")


# Determination of displacement and forces-------------------------------------
# creating all zero Q_values array with q elements
Q_valuess=np.empty([q], dtype=float)

for i in range(0,q,1):
    Q_valuess[i] = 0

Q=createMatrix(q,1,Q_valuess)

size_array_FEXT=FEXT.shape
row_FEXT=size_array_FEXT[0]
col_FEXT=size_array_FEXT[1]


for i in range(0,row_FEXT,1):
    joint_id = int(FEXT[i,0]-1)  # Joint number on which an external force is applied
    Cn = DOF[joint_id,:] # dof numbers of the node
    first_index = int(Cn[0]-1)
    second_index = int(Cn[1]-1)

    Q[first_index] = Q[first_index] + FEXT[i,1]    # assembly of Fy to Q
    Q[second_index] = Q[second_index] + FEXT[i,2]  # assembly of M to Q

#print(Q)

rf = displacement_counter   # rf: number of rows of Kff,displacement counter has defined in line 41

Kff = K[0:rf,0:rf]
Kfp = K[0:rf,rf:]
Kpf = K[rf:,0:rf]
Kpp = K[rf:,rf:]

#print("Kpf=",Kpf)

size_array_Cn=Cn.shape

row_Cn=1
col_Cn=size_array_Cn[0]


Fext = Q[0:rf]
Fext_1=Q[rf:q]
print("Fext",Fext)
#print(Fext)

# PRESCRIBED DISPLACEMENTS : ALL OF THEM ARE ZERO
Dp_values=np.empty([q-rf], dtype=float)

for i in range(0,q-rf,1):
    Dp_values[i] = 0

Dp=createMatrix(q-rf,1,Dp_values)

Df=calculate_Df(Kff,Fext,Kfp,Dp)

#print("Df=")
#print (Df)



len_Df=len(Df)
len_Dp = len(Dp)

D=np.empty([(len_Df+len_Dp),1], dtype=float)

counter_D_matrix=0
for i in range(0,len_Df,1):
    D[counter_D_matrix]=Df[i]
    counter_D_matrix = counter_D_matrix+1

for j in range(0,len_Dp,1):
    D[counter_D_matrix]=Dp[j]
    counter_D_matrix=counter_D_matrix+1

print("D=")
print (D)

R=calculate_R(Kpf,Df,Kpp,Dp)

print("R=")
print (R)

for i in range(0,row_NC,1):
    D_values=np.zeros(row_NC*col_NC, dtype=float)
    Qo_values=np.zeros(row_NC*col_NC, dtype=float)
    counter_Q=0
    counter_D=0
    for j in range(0,col_NC,1):
        row_DOF=int(NC[i,j])

        D1=int(DOF[row_DOF-1,0])
        D_values[counter_D]=D[D1-1]
        counter_D=counter_D+1

        D2=int(DOF[row_DOF-1,1])
        D_values[counter_D]=D[D2-1]
        counter_D=counter_D+1


        row_DOF=int(NC[i,j])

        Q1=int(DOF[row_DOF-1,0])
        Qo_values[counter_Q]=Q[Q1-1]
        counter_Q=counter_Q+1

        Q2=int(DOF[row_DOF-1,1])
        Qo_values[counter_Q]=Q[Q2-1]
        counter_Q=counter_Q+1

    #print(D_values)

    U = createMatrix(4,1,D_values)

    Qo = createMatrix(4,1,Qo_values)

    f_prime = calculate_memberEndForce_q(k_prime_add[i],T_trans[i],U)
    AVG_STRESS=(f_prime[0][0]-f_prime[1][0])/A
    #print("σavg",i+1,":",AVG_STRESS)
    print("fprime",(i+1),"=")
    print (f_prime)


#plotting undeformed(initial) shape----------------------------------
for i in range(0,row_NC):

    bn_id=int(NC[i,0]-1.0) # Joint number of the beginning joint of element
    en_id=int(NC[i,1]-1.0) # Joint number of the end joint of element

    x=np.array([[XY[bn_id,0]],[XY[en_id,0]]]) # x coord. of the beg. & end nodes in a row
    y=np.array([[XY[bn_id,1]],[XY[en_id,1]]]) # y coord. of the beg. & end nodes in a row


    plt.plot(x,y,'-b')
plt.xlim([-1.0, 7])
plt.ylim([-1.0, 5])
plt.show()

#plotting deformed shape----------------------------------
for i in range(0,row_NC):
    bn_id=int(NC[i,0]-1.0) # Joint number of the beginning joint of element
    en_id=int(NC[i,1]-1.0) # Joint number of the end joint of element

    x_DOF_bn = int(DOF[bn_id,0]-1.0) #x coord. of the beggining node
    x_DOF_en = int(DOF[en_id,0]-1.0) # x coord. of the end node
    y_DOF_bn = int(DOF[bn_id,1]-1.0) # y coord. of the beggining node
    y_DOF_en = int(DOF[en_id,1]-1.0)  # y coord. of the end node


    #if Dof number of any node bigger than row of Def ,Dp values must be added to
    # XY coordinates shown below
    # else Df values must be added to XY coordinates
    if x_DOF_bn > len_Df:
        Dp_value1 = Dp[int(x_DOF_bn-len_Df),0]  # determining index of Dp
        x_bn_value = XY[bn_id,0] + Dp_value1
    else:
        Df_value1 = Df[bn_id,0]
        x_bn_value = XY[bn_id,0]+ Df_value1


    if x_DOF_en > len_Df:
        Dp_value2 = Dp[int(x_DOF_en-len_Df),0]
        x_en_value = XY[en_id,0]+ Dp_value2
    else:
        Df_value2 = Df[en_id,0]
        x_en_value = XY[en_id,0]+ Df_value2


    if y_DOF_bn > len_Df:
        Dp_value3 = Dp[int(y_DOF_bn-len_Df),0]
        y_bn_value = XY[bn_id,1]+ Dp_value3
    else:
        Df_value3 = Df[bn_id,0]
        y_bn_value = XY[bn_id,1]+ Df_value3


    if y_DOF_en > len_Df:
        Dp_value4 = Dp[int(y_DOF_en-len_Df),0]
        y_en_value = XY[en_id,1]+ Dp_value4
    else:
        Df_value4 = Df[en_id,0]
        y_en_value = XY[en_id,1]+ Df_value4

    x_def = [x_bn_value,x_en_value] # x coord. of the beg. & end nodes in a row
    y_def = [y_bn_value,y_en_value] # y coord. of the beg. & end nodes in a row


    plt.plot(x_def,y_def,'--om','MarkerSize',10)
plt.show()



