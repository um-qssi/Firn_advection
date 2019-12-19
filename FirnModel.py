from __future__ import division
from fenics import *
#from dolfin import *
from numpy import *
from ufl import exp
from scipy import integrate

class FirnModel(object):
    
    def __init__(self,Ts,rhos,b,dt,ws):
        self.Ts=Ts
        self.rhos=rhos
        self.b=b
        self.dt=dt
        self.ws=ws
        
    def get_constants(self):
        self.rhoi=917.  #ice density......................kg/m^3
        self.cpi=2100. # heat capacity of ice ........... J/(kg K)
        self.R=8.3144621  #Gas cosntant.................. J/(mol K)
        self.spy=365*24*60*60.  ###seconds per minute
        self.Lf=334000.   #Latent heat of fusion..........J/(kg)
        self.pw=1000.  #Density of water................kg/(m^3)

    def initialize_mesh(self,n,zs,zb):
        mesh=IntervalMesh(n,zb,zs)
        z = mesh.coordinates()[:,0]  
        z=z[::-1]*-1
        
        V=FunctionSpace(mesh,FiniteElement('CG',mesh.ufl_cell(), 1))
        self.rho = Function(V)  #density
        self.w = Function(V)  #vertical vel
        self.T = Function(V)  #temperature
        
        hf=-zb/n  #height of individual firn package
        
        self.hf=hf
        self.zs=zs
        self.V=V     
        self.n=n
        self.zb=zb
        self.z=z
        
    def initial_conditions(self):
        V=self.V
        Ts=self.Ts #Temperature at surface
        ws=self.ws #vertical velocity at surface
        rhos=self.rhos #Density at surface
        zs=self.zs  #surface
        T_init=Constant(Ts)
        w_init=Constant(ws)
        rho_init=Constant(rhos)
        
        #assign previous value based on initial conditions  
        self.rho_p=interpolate(rho_init,V)
        self.w_p = interpolate(w_init,V)
        self.T_p= interpolate(T_init,V)

        def boundary(x, on_boundary):
            return on_boundary and x[0]==zs
        
        #boundary conditions
        self.bc_rho = DirichletBC(V, rho_init, boundary)
        self.bc_w = DirichletBC(V, w_init, boundary)
        self.bc_T = DirichletBC(V, T_init, boundary)
        
    def hl(self):
        ##HL method with no melt (Herron & Langway, 1980)
        b=self.b
        R=self.R
        rhoi=self.rhoi
        dt=self.dt
        cpi=self.cpi    
        spy=self.spy
        
        V=self.V        
        phi = TestFunction(V)
        #latent heat
        Se=Function(V)

        dr=TrialFunction(V)
        dw=TrialFunction(V)
        dte=TrialFunction(V)
        
        rho=self.rho
        w=self.w
        T=self.T       
        
        rho_p=self.rho_p
        w_p=self.w_p
        T_p=self.T_p

        bc_rho=self.bc_rho
        bc_w=self.bc_w
        bc_T=self.bc_T
        
        c0=11*(b*0.917)*exp(-10160/(R*T))

        c1=575*(b*0.917)**0.5*exp(-10160/(R*T))
        
        c=conditional(550>rho_p,c0,c1)
                
        lb = interpolate(Constant(300),V)   # lower
        ub = interpolate(Constant(1000),V) 
        lb1 = interpolate(Constant(-500),V)   # lower
        ub1 = interpolate(Constant(500),V) 
        lb2 = interpolate(Constant(200),V)   # lower
        ub2 = interpolate(Constant(300),V) 
        
        #Vertical Velocity 
        F_w=w.dx(0)*rho*phi*dx+c*(rhoi-rho)*phi*dx  
    
        #Density
        F=(rho-rho_p)/(dt/spy)*phi*dx-c*(rhoi-rho)*phi*dx+w*rho.dx(0)*phi*dx
    
        #Temperature
        F_T=rho*cpi*(T-T_p)/dt*phi*dx+2.1*(rho/rhoi)**2*dot(T.dx(0),phi.dx(0))*dx\
            -4.2*rho/(rhoi**2)*inner(rho.dx(0),T.dx(0))*phi*dx+4.2/(rhoi**2)*5.6e-2*2.7**(5.7e-3*260.)*inner(T.dx(0),phi.dx(0))*dx+rho*cpi*w/spy*T.dx(0)*phi*dx-Se*phi*dx       
                    
        J=derivative(F,rho,dr)  
        J1=derivative(F_w,w,dw)  
        J2=derivative(F_T,T,dte)

        problem=NonlinearVariationalProblem(F,rho,bcs=bc_rho,J=J)
        problem.set_bounds(lb,ub)
        
        problem2=NonlinearVariationalProblem(F_w,T,bcs=bc_w,J=J1)
        problem2.set_bounds(lb1,ub1)
    
        problem3=NonlinearVariationalProblem(F_T,T,bcs=bc_T,J=J2)
        problem3.set_bounds(lb2,ub2)
 
        solver=NonlinearVariationalSolver(problem)
        solver2=NonlinearVariationalSolver(problem2)
        solver3=NonlinearVariationalSolver(problem3)
        
        def set_solver_options(nls):
            nls.parameters['nonlinear_solver'] = 'snes'
            nls.parameters['snes_solver']['method'] = 'vinewtonrsls'
            nls.parameters['snes_solver']['error_on_nonconvergence'] = True
            nls.parameters['snes_solver']['linear_solver'] = 'mumps'
            nls.parameters['snes_solver']['maximum_iterations'] = 100
            nls.parameters['snes_solver']['report'] = False   
                
        set_solver_options(solver) 
        set_solver_options(solver2)
        set_solver_options(solver3)
                
        self.phi=phi
        
        self.F=F
        self.J=J
        
        self.F_T=F_T
        self.J2=J2
        
        self.F_w=F_w
         
        self.solver=solver
        self.solver2=solver2
        self.solver3=solver3
        
        self.rho=rho
        self.w=w
        self.T=T
        self.Se=Se
        
    def reeh_melt(self,melt):
        #(Reeh et al., 2005)
        b=self.b    #accumulation
        rhos=self.rhos  #surface density
        rhoi=self.rhoi  #ice density
        spy=self.spy  #seconds per minute
        Se=self.Se #latent heat
        Lf=self.Lf #latent heat of fusion
        V=self.V #mesh
        zs=self.zs #surface
        F=self.F #density equation
        J=self.J #Jacobian
        rho=self.rho #density
        pw=self.pw #density of water
        dt=self.dt #delta_time
        hf=self.hf #height of firn
        
        def boundary(x, on_boundary):
            return on_boundary and x[0]==zs   
        
        #average firn density at surface with melt 
        denominator=1-melt/b*(1-rhos/rhoi)
        rho_theta=rhos/denominator

        #update boundary condition
        bc_rho=DirichletBC(V,Constant(rho_theta),boundary)
        
        #update  vertical velocity boundary condition
        vel=-b*(rhoi/rho_theta)
        bc_w=DirichletBC(V,Constant(vel),boundary) 
        
        #latent heat
        latent_heat=Lf*(b/(rhos/1000.))*pw*(melt/b)/dt  
        Se.vector()[0]=latent_heat
        
        lb = interpolate(Constant(300),V)   # lower
        ub = interpolate(Constant(1000),V) 
        
        problem=NonlinearVariationalProblem(F,rho,bcs=bc_rho,J=J)
        problem.set_bounds(lb,ub)
        solver=NonlinearVariationalSolver(problem)
        
        def set_solver_options(nls):
            nls.parameters['nonlinear_solver'] = 'snes'
            nls.parameters['snes_solver']['method'] = 'vinewtonrsls'
            nls.parameters['snes_solver']['error_on_nonconvergence'] = True
            nls.parameters['snes_solver']['linear_solver'] = 'mumps'
            nls.parameters['snes_solver']['maximum_iterations'] = 100
            nls.parameters['snes_solver']['report'] = False   
 
        set_solver_options(solver)
        
        self.bc_rho=bc_rho
        self.solver=solver
        self.bc_w=bc_w
        self.Se=Se        
        
    def tippingbucket(self,melt):
        dt=self.dt  #time step
        rho=self.rho #density
        b=self.b     #accumulation
        rhos=self.rhos #surface density
        rhoi=self.rhoi #ice density
        hf=self.hf   #height of individual firn column
        z=self.z  #vertical coordinates
        T=self.T  #Temperature
        w=self.w  #vertical velocity
        cpi=self.cpi  #heat capacity
        Lf=self.Lf  #Latent heat of fusion
        pw=self.pw  #Density of water
        spy=self.spy #seconds per year
        zs=self.zs   #surface     
        n=self.n  #number of firn layers
        Se=self.Se #Latent heat
        V=self.V  #FunctionSpace
        F=self.F  #Density equation
        J=self.J  #Jacobian for Density

        def boundary(x, on_boundary):
            return on_boundary and x[0]==zs  
        
        #convert to water equivalent
        water=melt*0.917
        b=b*0.917
        
        #initialize water content temperature and density
        water_content=zeros(n)
        Temperature=T.vector().get_local()
       
        dens=rho.vector().get_local()
        density_final=append(rhos,dens)
        density=density_final[0:-1]
        
        rho_firnice=[]
        water_firnice=[]
        
        #loop through layers
        #####################################
        for i in range(n):
            #calculate porosity, irreducible water content, and water holding content
            porosity=1-density[i]/rhoi
            Irreducible_Water=1.7+5.7*(porosity/(1-porosity))
            WHC=(density[i]*hf*Irreducible_Water/100.)/(pw-pw*Irreducible_Water/100.)
            
            #Cold content
            RC=density[i]*cpi*hf*(273.16-Temperature[i])/(pw*Lf)
            
            #If density is greater than 830 kg/m^3 water runs off, if cold content is greater than
            #amount of water all water stays in layer, if coldent content plus with holding content
            #is greater than water, less than the whc is kept, else cold content is frozen water holding
            #content is WHC
            if density[i]>=830:
                water_content[i]=0
                refreezing=0
            elif water<=RC:
                water_content[i]=0
                refreezing=water
            elif water<=RC+WHC:
                refreezing=RC
                water_content[i]=water-RC
            else:
                refreezing=RC
                water_content[i]=WHC
            
            #update water content and store water
            water=water-(water_content[i]+refreezing)
            water_firnice.append(water)
            
            #fraction ice 
            melt_fraction=(pw*(refreezing+water_content[i]))/(pw*(refreezing+water_content[i])+density[i]*hf)
            #updated density 
            rho_theta=density[i]*(1-melt_fraction)+melt_fraction*rhoi
            
            rho.vector()[i]=rho_theta
            rho_firnice.append(rho_theta)
            
            #latent heat
            latent_heat=Lf*hf*pw*melt_fraction/dt
            Se.vector()[i]=latent_heat
        
        
        ###New surface boundary conditions
        bw=(b-water_firnice[0])/0.917   ##iceequiv added to the surface
        w_s=Constant(-(bw)*rhoi/rho_firnice[0])
        
        bc_w=DirichletBC(V,w_s,boundary)
        bc_rho=DirichletBC(V,Constant(rho_firnice[0]),boundary)
            
        lb = interpolate(Constant(300),V)   # lower
        ub = interpolate(Constant(1000),V) 
        
        problem=NonlinearVariationalProblem(F,rho,bcs=bc_rho,J=J)
        problem.set_bounds(lb,ub)
        solver=NonlinearVariationalSolver(problem)
        
        def set_solver_options(nls):
            nls.parameters['nonlinear_solver'] = 'snes'
            nls.parameters['snes_solver']['method'] = 'vinewtonrsls'
            nls.parameters['snes_solver']['error_on_nonconvergence'] = True
            nls.parameters['snes_solver']['linear_solver'] = 'mumps'
            nls.parameters['snes_solver']['maximum_iterations'] = 100
            nls.parameters['snes_solver']['report'] = False   
 
        set_solver_options(solver)
                
        self.solver=solver     
            
        self.rho_firnice=rho_firnice   
        self.Se=Se
        self.bc_w=bc_w
        self.bc_rho=bc_rho
        self.rho=rho        
            
    def update_inputs(self,Tnew,bnew):
        ##updates inputs to the model when surface Temperature and accumulation changes           
        V=self.V #FunctionSpace
        rhoi=self.rhoi #ice Density
        rhos=self.rhos #surface firn density
        zs=self.zs  #surface
        F_T=self.F_T #temperature equation
        J2=self.J2  #Temperature Jacobian
        T=self.T  #Temperature
        R=self.R  #Gas Constant
        rho_p=self.rho_p  #previous density
        dt=self.dt  ##time step
        spy=self.spy #seconds per year
        w=self.w  #vertical velocity
        bc_rho=self.bc_rho  #boundary condition density
        rho=self.rho #density
        phi=self.phi #test function
        J=self.J ##density jacobian

        def boundary(x, on_boundary):
            return on_boundary and x[0]==zs   
        
        #set new boundary conditions
        bc_T=DirichletBC(V,Constant(Tnew),boundary)
    
        w_s=Constant(-bnew*rhoi/rhos)
        bc_w=DirichletBC(V,w_s,boundary) 
        
        #set equations
        c0=11*(bnew*0.917)*exp(-10160/(R*T))
        c1=575*(bnew*0.917)**0.5*exp(-21400/(R*T))
        
        c=conditional(550>rho_p,c0,c1)
            
        F=(rho-rho_p)/(dt/spy)*phi*dx-c*(rhoi-rho)*phi*dx+w*rho.dx(0)*phi*dx
        F_w=w.dx(0)*rho*phi*dx+c*(rhoi-rho)*phi*dx
        
        self.bc_w=bc_w
        self.b=bnew
        self.F_w=F_w
        self.F=F        

        lb2 = interpolate(Constant(200),V)   
        ub2 = interpolate(Constant(300),V) 
       
        problem3=NonlinearVariationalProblem(F_T,T,bcs=bc_T,J=J2)
        problem3.set_bounds(lb2,ub2)

        
        solver3=NonlinearVariationalSolver(problem3)

        def set_solver_options(nls):
            nls.parameters['nonlinear_solver'] = 'snes'
            nls.parameters['snes_solver']['method'] = 'vinewtonrsls'
            nls.parameters['snes_solver']['error_on_nonconvergence'] = True
            nls.parameters['snes_solver']['linear_solver'] = 'mumps'
            nls.parameters['snes_solver']['maximum_iterations'] = 100
            nls.parameters['snes_solver']['report'] = False   
 
        set_solver_options(solver3)
               
        lb = interpolate(Constant(300),V)   # lower
        ub = interpolate(Constant(1000),V) 
        
        problem=NonlinearVariationalProblem(F,rho,bcs=bc_rho,J=J)
        problem.set_bounds(lb,ub)
        solver=NonlinearVariationalSolver(problem)
        set_solver_options(solver)
                
        self.solver=solver
        self.solver3=solver3
        self.bc_T=bc_T
        
    def take_step(self):
        #take time step
        solver=self.solver
        solver2=self.solver2
        solver3=self.solver3
        
        rho_p=self.rho_p
        w_p=self.w_p
        T_p=self.T_p
        
        rho=self.rho
        w=self.w
        T=self.T
        
        F_w=self.F_w
        bc_w=self.bc_w

        solver.solve()
#        solver2.solve()
        solver3.solve()
        solve(F_w==0,w,bc_w)
    
        rho_p.assign(rho)
        w_p.assign(w)
        T_p.assign(T)    

        self.rho=rho
        self.w=w
        self.T=T
        
    def calc_pco(self):
        ##calculate pore close off
        rho=self.rho #density
        z=self.z     #vertical depth    
        
        density=rho.vector().get_local()

        density=density[density<830]
        firn_index=len(density)-1
        pco=z[firn_index]
        
        firn_z=z[z<=pco]
        
        self.density_pco=density
        self.z12=z12
        self.firn_index=firn_index
        return pco
        
    def calc_temp_at_pco(self):
        T=self.T
        firn_index=self.firn_index
        
        temp=T.vector().get_local()
        temp_pco=temp[firn_index]
 
        return temp_pco
       
    def calc_capacity(self):
        density=self.density_pco
        firn_z=self.firn_z
        
        firn_load=integrate.trapz(density,firn_z)
        pure_ice=843*ones(len(density))
        capacity=integrate.trapz(pure_ice,firn_z)-firn_load
    
        return capacity
        
        
    def calc_air_content(self):
        density=self.density_pco
        firn_z=self.firn_z
        
        infiltration_ice_density=843
        
        firn_load=integrate.trapz(density,firn_z)
        pure_ice=843*ones(len(density))
        capacity=integrate.trapz(pure_ice,firn_z)-firn_load
        air_content=capacity/infiltration_ice_density
        
        return air_content
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        