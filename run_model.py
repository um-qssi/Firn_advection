from FirnModel import *
import matplotlib.pyplot as plt

Ts = 254.3 ####surface temperature (K)
b = 0.3   ###accumulation (ice equivalent m/a)
rhos = 360  ###initial density (kg/m3)
spy=365*24*60*60.  ##seconds per year

dt=1*spy  ##time-step (s)

zs=0    ##surface start
zb=-80   ##surface end
n=zb*-2   ##number of z-positions


model=FirnModel(Ts,rhos,b,dt)

model.initialize_mesh(n,zs,zb)
model.get_constants()
model.initial_conditions()

model.hl()

model.take_step()

Time=100*spy
t=0

while t<Time:    
    t = t+dt
    Ts = Ts + 0.01
    b = b + 0.01
    
    model.update_inputs(Ts,b)
    model.take_step()
    
    

######plot data
density=model.rho.vector().get_local()
temperature=model.T.vector().get_local()

depth=model.z

plt.figure()
plt.plot(density,depth)
plt.gca().invert_yaxis()
plt.xlabel('Density',fontsize=16)
plt.ylabel('Depth',fontsize=16)
plt.show()

plt.figure()
plt.plot(temperature,depth)
plt.gca().invert_yaxis()
plt.xlabel('Temperature',fontsize=16)
plt.ylabel('Depth',fontsize=16)
