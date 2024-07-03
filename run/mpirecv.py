import os
from prec_automl import pred_automl
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
#split the communicator based on rank
color = 0
key = comm.Get_rank()
newcomm = comm.Split(color, key)
rank = comm.Get_rank()

tag=0
# if rank == 31:
logfile="log"+str(rank)+".txt"
logf=open(logfile,"w")
# time_count=10
# while time_count>1:
while True:
    status = MPI.Status()
    # while True:
    #not sure which source will send data to me
    comm.Probe(source=MPI.ANY_SOURCE, tag=tag, status=status)
    lenth = status.Get_elements(MPI.REAL8)
    # logf.write('Received lenth:'+str(lenth)+"\n")
    spaceship = np.empty(lenth, dtype=np.double)
    comm.Recv([spaceship, MPI.REAL8], source=status.Get_source(), tag=tag)
    spaceship=np.reshape(spaceship,(int(spaceship.shape[0]/12),12))
    t2m_f=spaceship[:,1]
    sp_f=spaceship[:,2]
    q_f=spaceship[:,3]
    strd_f=spaceship[:,4]
    ssrd_f=spaceship[:,5]
    ws_f=np.sqrt(spaceship[:,6]**2+spaceship[:,7]**2)
    elev_f=spaceship[:,0]
    julian_day=spaceship[:,8]
    lat=spaceship[:,9]
    lon=spaceship[:,10]
    x = np.array([t2m_f,sp_f,q_f,strd_f,ssrd_f,ws_f,julian_day,lat,lon]).T#.reshape(1, -1)
    prec=np.array(pred_automl(x), dtype=np.float64)
    logf.write('Received temp:'+str(x)+" "+str(prec)+"\n")
    comm.Send([prec, MPI.REAL8], dest=status.Get_source(), tag=tag)
    # time_count-=1
logf.close()
exit()

