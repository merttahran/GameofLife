from mpi4py import MPI
import numpy as np
import sys
import math
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
c =  int(math.sqrt(size-1))
dim=int(360/c)#dimension of the block for which a process responsible
#master process
if rank ==0:
    turn = np.empty(1,dtype=int)
    turn[0] = int(sys.argv[3])
    #input part
    input = open(sys.argv[1],"r");
    matrix = [[int(digit) for digit in line.split()] for line in input]
    input.close()
    matrix = np.array(matrix)
    #broadcast the submatrix to a servant process
    for vertical in range(0,c):#row order of a block
        for horizontal in range(0,c): #column order of a block
            submatrix= matrix[vertical*dim:vertical*dim+dim, horizontal*dim:horizontal*dim+dim]
            submatrix = np.array(submatrix)
            order = vertical*c+horizontal+1
            comm.Send(turn, dest=order, tag=21)
            comm.Send(submatrix, dest=order, tag=11)
    #collect the final states of the map
    tempMatrix = np.array(np.empty([dim,dim],dtype=int))
    comm.Recv(tempMatrix, source=1, tag=23)
    for horizontal in range(1,c):
            submatrix = np.empty([dim,dim],dtype=int)
            submatrix = np.array(submatrix)
            order = horizontal+1
            comm.Recv(submatrix, source=order, tag=23)
            tempMatrix = np.concatenate((tempMatrix,submatrix),axis=1)
    matrix=tempMatrix
    for vertical in range(1,c):
        tempMatrix = np.array(np.empty([dim,dim],dtype=int))
        comm.Recv(tempMatrix, source=vertical*c+1, tag=23)
        for horizontal in range(1,c):
            submatrix= np.empty([dim,dim],dtype=int)
            submatrix = np.array(submatrix)
            order = vertical*c+horizontal+1
            comm.Recv(submatrix, source=order, tag=23)
            tempMatrix = np.concatenate((tempMatrix,submatrix),axis=1)
        matrix=np.concatenate((matrix,tempMatrix))
    #output part
    np.savetxt(sys.argv[2],matrix,newline=' \n',fmt='%d')
else:
    #servant process
    turn = np.empty(1,dtype=int)
    comm.Recv(turn, source=0, tag=21)
    turn = turn[0]
    matrix = np.empty(dim*dim, dtype=int)
    tempMatrix = np.empty(1, dtype=int)
    comm.Recv(matrix, source=0, tag=11)
    matrix=matrix.reshape(dim,dim);
    myRank = (rank+size -1) %size#make calculations of locations of neighbours easier for me
    upper=(myRank+size-c-1)%(size-1)+1
    lower=(myRank+c)%(size-1)+1
    left = myRank//c*c+(myRank-1+c)%c+1
    right = myRank//c*c+(myRank+1)%c+1
    # create empty arrays to information from the neighbors
    up = np.empty(dim,dtype=int)
    up=up.reshape(1,dim)
    down = np.empty(dim,dtype=int)
    down=down.reshape(1,dim)
    nonleft = np.empty(dim+2,dtype=int)
    nonright = np.empty(dim+2,dtype=int)
    nonleft=nonleft.reshape(dim+2,1)
    nonright=nonright.reshape(dim+2,1)
    
    if turn > 0:
        for x in range(turn):
            if  (rank//c)%2==0:#check the row order
                comm.Recv(down,source=lower,tag=13)
                comm.Send(matrix[0], dest=upper, tag=13)
                comm.Recv(up,source=upper,tag=14)
                comm.Send(matrix[-1], dest=lower, tag=14) 
                tempMatrix = np.concatenate((up,matrix,down))
            else:   
                comm.Send(matrix[0], dest=upper, tag=13)
                comm.Recv(down,source=lower,tag=13)
                comm.Send(matrix[-1], dest=lower, tag=14)
                comm.Recv(up,source=upper,tag=14)
                tempMatrix = np.concatenate((up,matrix,down))
            # at this point, each block has the upper row and lower row of the map
            # therefore, each block can send this information to its own left and right neighbors
            if rank%2==0:
                comm.Send(np.array(tempMatrix[:,0]), dest=left, tag=15)
                comm.Recv(nonleft,source=right,tag=15)
                comm.Send(np.array(tempMatrix[:,-1]), dest=right, tag=16)
                comm.Recv(nonright,source=left,tag=16)
                tempMatrix = np.concatenate((nonright,tempMatrix,nonleft),axis=1)
            else:
                comm.Recv(nonleft,source=right,tag=15)
                comm.Send(np.array(tempMatrix[:,0]), dest=left, tag=15)
                comm.Recv(nonright,source=left,tag=16)
                comm.Send(np.array(tempMatrix[:,-1]), dest=right, tag=16)
                tempMatrix = np.concatenate((nonright,tempMatrix,nonleft),axis=1)
            #extended map is available and computation begins
            for x in range(dim):
                for y in range(dim):
                    sum =    tempMatrix[x,y]+tempMatrix[x,y+1]+tempMatrix[x,y+2]+tempMatrix[x+1,y]+tempMatrix[x+1,y+2]+tempMatrix[x+2,y]+tempMatrix[x+2,y+1]+tempMatrix[x+2,y+2]
                    #sum is the number of neighbours
                    if matrix[x][y] ==1:#check if a creature is there
                        if sum < 4 and sum >1:
                            matrix[x][y]=1
                        else:#overpopulation or loneliness
                            matrix[x][y]=0
                    else:
                        if sum ==3:#regeneration
                            matrix[x][y]=1
    #send the last state of the block to the master process
    comm.Send(matrix, dest=0, tag=23)
