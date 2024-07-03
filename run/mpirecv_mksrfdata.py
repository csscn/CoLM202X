import os
from mpi4py import MPI
import numpy as np
from trainer import train

comm = MPI.COMM_WORLD
#split the communicator based on rank
color = 0
key = comm.Get_rank()
newcomm = comm.Split(color, key)
rank = comm.Get_rank()
tag = 0
# if rank == 31:
logfile="log_domain"+str(rank)+".txt"
logf=open(logfile,"w")
# time_count=10
# while time_count>1:
status = MPI.Status()
# while True:
#not sure which source will send data to me
comm.Probe(source=MPI.ANY_SOURCE, tag=tag, status=status)
lenth = status.Get_elements(MPI.REAL8)
# logf.write('Received lenth:'+str(lenth)+"\n")
domain = np.empty(lenth, dtype=np.double)
status = MPI.Status()
comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=status)
str_length = status.Get_elements(MPI.CHAR)
fprefix = bytearray(str_length)
complete = "complete!"
comm.Recv(domain, source=status.Get_source(), tag=tag)
comm.Recv([fprefix,MPI.CHAR], source=status.Get_source(), tag=1)
# Decode the bytearray to a string
fprefix = fprefix.decode('utf-8')
lat_s = domain[0]
lat_n = domain[1]
lon_w = domain[2]
lon_e = domain[3]
start_y = int(domain[4])
start_m = domain[5]
end_y = int(domain[6])
end_m = domain[7]
dir_forcing = {"t2m":{},"q":{},"sp":{},"prec":{},"u10":{},"v10":{},"sw":{},"lw":{}}
vname = {"t2m":{},"q":{},"sp":{},"prec":{},"u10":{},"v10":{},"sw":{},"lw":{}}
var_dict = ["t2m","q","sp","prec","u10","v10","sw","lw"]
for i in range(8):
    dir_forcing[var_dict[i]] = fprefix[i*250:(i+1)*250].replace(" ","")
count = 8*250
for i in range(8):
    vname[var_dict[i]] = fprefix[i*250+count:(i+1)*250+count].replace(" ","")
count = 16*250
groupby = fprefix[count]
print('dir_forcing: \n',dir_forcing)
print('vname: \n',vname)
for year in range(start_y,end_y+1,1):
    train(year, lat_s, lat_n,lon_w, lon_e,"automl",dir_forcing,vname)
comm.Send([complete, MPI.CHARACTER], dest=status.Get_source(), tag=tag)
