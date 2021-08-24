# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2D Uplift model
#
#
# <table><tr><td><img src='./images/Uplift-0.png'></td><td><img src='./images/Uplift-56.png'></td></tr></table>
# (Left) Initial model setup. (Right) Model with steady state topography induced by basal traction condition (Stokes system's neumann condition).
#
# #### This model utilises scaling to enable model input in dimensional units but model computation and output in scaled units.

# %%
import numpy as np
import underworld as uw
import math
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise, non_dimensionalise as nd
import underworld.visualisation as vis
import os

model_end_step   = 3
output_path      = 'uplift/'
elType           = 'Q1/dQ0'
nEls             = (560,340)
# nEls             = (280,170)
# nEls             = (140,80)
box_ds           = [(0*u.km, 2800*u.km), (-700*u.km, 150*u.km)]
    
# search and build, if required, an output path
if uw.mpi.rank==0:
    try:
        if not os.path.exists("./"+output_path+"/"):
            os.makedirs("./"+output_path+'/')
    except:
        raise

# %%
# build reference units
KL          = 1     * u.km
K_density   = 1e3   * u.kg / (u.m)**3
K_viscosity = 1e19  * u.Pa * u.s

# KL          = 1.  * u.meter
# K_viscosity = 1.  * u.pascal * u.second
# K_density   = 1.  * u.kilogram / (u.meter)**3

KM = K_density * KL**3
KT = KM / (KL * K_viscosity)

# %%
# K_force     = 100  * u.kg * u.m / u.s**2
# KL          = 100  * u.km
# K_density   = 3e3  * u.kg / u.m**3

# K_force     = 1  * u.kg * u.m / u.s**2
# KL          = 1e3 * u.m
# K_density   = 1.  * u.kg / u.m**3

# KM = K_density * KL**3
# KT = np.sqrt(KM * KL / K_force)


# %%
K_substance  = 1. * u.mole
Kt_degrees   = 1. * u.kelvin


scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"]      = KL.to_base_units()
scaling_coefficients["[temperature]"] = Kt_degrees.to_base_units()
scaling_coefficients["[time]"]        = KT.to_base_units()
scaling_coefficients["[mass]"]        = KM.to_base_units()

# %%
print("{:3e}, {:3e}".format(nd(1e19*u.Pa*u.s),nd(9.8*u.m*u.s**-2)))

# %%
# build mesh and mesh variables
mesh = uw.mesh.FeMesh_Cartesian( elementType = elType, 
                                 elementRes  = nEls, 
                                 minCoord    = [nd(box_ds[0][0]), nd(box_ds[1][0])], 
                                 maxCoord    = [nd(box_ds[0][1]), nd(box_ds[1][1])] )

bottomWall = mesh.specialSets["Bottom_VertexSet"]
topWall    = mesh.specialSets["Top_VertexSet"]
iWalls     = mesh.specialSets["Left_VertexSet"] + mesh.specialSets["Right_VertexSet"]
jWalls     = mesh.specialSets["Top_VertexSet"] + mesh.specialSets["Bottom_VertexSet"]

velocityField = mesh.add_variable( nodeDofCount=mesh.dim )
pressureField = mesh.subMesh.add_variable( nodeDofCount=1 )

# %%
# nd mid-point
mid = 0.5 *(np.array(mesh.maxCoord)-np.array(mesh.minCoord))
plume_x = [mid[0], nd(box_ds[1][0]+300*u.km)]
plume_r = nd(50*u.km)

dt = nd(15*u.kyr)


# %%
# create checkpoint function
def checkpoint( mesh, fieldDict, swarm, swarmDict, index,
                meshName='mesh', swarmName='swarm', time=None, 
                prefix='./', enable_xdmf=True):
    import os
    # Check the prefix is valid
    if prefix is not None:
        if not prefix.endswith('/'): prefix += '/' # add a backslash
        if not os.path.exists(prefix) and uw.mpi.rank==0:
            print("Creating directory: ",prefix)
            os.makedirs(prefix)
        uw.mpi.barrier() 
       
    # initialise internal time
    if time is None and not hasattr(checkpoint, '_time'):
        checkpoint.time = 0
    # use internal time
    if time is None:
        time = checkpoint.time + 1
    
    if not isinstance(index, int):
        raise TypeError("'index' is not of type int")        
    ii = str(index)
    
    if mesh is not None:
        
        # Error check the mesh and fields
        if not isinstance(mesh, uw.mesh.FeMesh):
            raise TypeError("'mesh' is not of type uw.mesh.FeMesh")
        if not isinstance(fieldDict, dict):
            raise TypeError("'fieldDict' is not of type dict")
        for key, value in fieldDict.items():
            if not isinstance( value, uw.mesh.MeshVariable ):
                raise TypeError("'fieldDict' must contain uw.mesh.MeshVariable elements")


        # see if we have already saved the mesh. It only needs to be saved once
        if not hasattr( checkpoint, 'mH' ):
            checkpoint.mH = mesh.save(prefix+meshName+".h5")
        mh = checkpoint.mH

        for key,value in fieldDict.items():
            filename = prefix+key+'-'+ii
            handle = value.save(filename+'.h5')
            if enable_xdmf: value.xdmf(filename, handle, key, mh, meshName, modeltime=time)
        
    # is there a swarm
    if swarm is not None:
        
        # Error check the swarms
        if not isinstance(swarm, uw.swarm.Swarm):
            raise TypeError("'swarm' is not of type uw.swarm.Swarm")
        if not isinstance(swarmDict, dict):
            raise TypeError("'swarmDict' is not of type dict")
        for key, value in swarmDict.items():
            if not isinstance( value, uw.swarm.SwarmVariable ):
                raise TypeError("'fieldDict' must contain uw.swarm.SwarmVariable elements")
    
        sH = swarm.save(prefix+swarmName+"-"+ii+".h5")
        for key,value in swarmDict.items():
            filename = prefix+key+'-'+ii
            handle = value.save(filename+'.h5')
            if enable_xdmf: value.xdmf(filename, handle, key, sH, swarmName, modeltime=time)

# %%
# # visualise the bottom stress condition
# if uw.mpi.size == 1:
#     uw.utils.matplotlib_inline()
#     import matplotlib.pyplot as pyplot
#     import matplotlib.pylab as pylab
#     pyplot.ion()
#     pylab.rcParams[ 'figure.figsize'] = 12, 6
#     pyplot.title('Prescribed traction component normal to base wall')
#     km_scaling  = dimensionalise(1,u.kilometer)
#     MPa_scaling = dimensionalise(1,u.MPa)
#     pyplot.xlabel('X coordinate - (x{}km)'.format(km_scaling.magnitude))
#     pyplot.ylabel('Normal basal traction MPa - (x{:.3e}MPa)'.format(MPa_scaling.magnitude))
    
#     xcoord = mesh.data[bottomWall.data][:,0]          # x coordinate
#     stress = tractionField.data[bottomWall.data][:,1] # 2nd component of the traction along the bottom wall
    
#     pyplot.plot( xcoord, stress, 'o', color = 'black', label='numerical') 
#     pyplot.show()

# %%
# Initialise a swarm.
swarm = uw.swarm.Swarm( mesh=mesh, particleEscape=True )
advector= uw.systems.SwarmAdvector(velocityField, swarm, order=2)

# Add a data variable which will store an index to determine material.
materialVariable = swarm.add_variable( dataType="double", count=1 )

# Create a layout object that will populate the swarm across the whole domain.
swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
# Populate.
swarm.populate_using_layout( layout=swarmLayout )

# material 0 - air
# material 1 - lithosphere
# material 2 - mantle
# material 4 - plume

# air_props    = [nd(3200*u.kg/u.m**-3), nd(1e19*u.Pa*u.s), 0]
# litho_props  = [nd(3300*u.kg/u.m**-3), nd(1e23*u.Pa*u.s), 1]
# mantle_props = [nd(3300*u.kg/u.m**-3), nd(1e21*u.Pa*u.s), 2]
# plume_props  = [nd(3200*u.kg/u.m**-3), nd(1e20*u.Pa*u.s), 3]

air_props    = [nd(   0*u.kg/u.m**-3), nd(1e19*u.Pa*u.s), 0]
litho_props  = [nd(3300*u.kg/u.m**-3), nd(1e23*u.Pa*u.s), 1]
mantle_props = [nd(3300*u.kg/u.m**-3), nd(1e21*u.Pa*u.s), 2]
plume_props  = [nd(3200*u.kg/u.m**-3), nd(1e20*u.Pa*u.s), 3]

# air_props    = [nd(32*u.kg/u.m**-3), nd(19*u.Pa*u.s), 0]
# litho_props  = [nd(33*u.kg/u.m**-3), nd(23*u.Pa*u.s), 1]
# mantle_props = [nd(33*u.kg/u.m**-3), nd(21*u.Pa*u.s), 2]
# plume_props  = [nd(32*u.kg/u.m**-3), nd(20*u.Pa*u.s), 3]

def easy_map(po, pi=-1):
    # returns a mapping dictionary for uw branching functions.
    #  pi is assumed to be the materialindex position, ie -1
    #  po is the desired (objective) property position
    return {  
             air_props[pi]: air_props[po],
             litho_props[pi]: litho_props[po],
             mantle_props[pi]: mantle_props[po],
             plume_props[pi]: plume_props[po] 
           }

H_surf = nd(0*u.km)
H_LAB = nd(-100*u.km)
H_ma  = nd(-700*u.km)
H_plume = nd(100*u.km)

materialVariable.data[:]=0
for index,coord in enumerate(swarm.particleCoordinates.data):
    if coord[1] > H_surf:
        materialVariable.data[index]=air_props[-1]
    elif coord[1] > H_LAB:
        materialVariable.data[index]=litho_props[-1]
    elif coord[1] > H_ma:
        materialVariable.data[index]=mantle_props[-1]
    
    inside = np.sqrt((coord[0]-plume_x[0])**2 + (coord[1]-plume_x[1])**2)
    if inside < plume_r:
        materialVariable.data[index]=plume_props[-1]

# population control regulars particle creation and deletion
# important for inflow/outflow problems
population_control = uw.swarm.PopulationControl(swarm, 
                                                aggressive=True,splitThreshold=0.15, maxDeletions=2,maxSplits=10,
                                                particlesPerCell=20)

# build tracer swarm for fluid level - only 1 particle
mswarm = uw.swarm.Swarm( mesh=mesh, particleEscape=True )
msAdvector= uw.systems.SwarmAdvector(velocityField, mswarm, order=2)

# initial height at 'air' level
particleCoordinates = np.zeros((1,2))
particleCoordinates[:,0] = mid[0]
particleCoordinates[:,1] = H_surf
ignore=mswarm.add_particles_with_coordinates(particleCoordinates)

# parallel safe way of finding the particles vertical coordinate.
fn_y = fn.coord()[1]
fn_y_minmax = fn.view.min_max(fn_y)

# %%
densityFn = uw.function.branching.map( fn_key   = materialVariable, 
                                       mapping  = easy_map(0) )

viscosityFn = uw.function.branching.map( fn_key  = materialVariable, 
                                         mapping = easy_map(1)  )
gravity = nd(10 * u.m/u.s**2)
forceFn = densityFn * (0.0,-gravity)

# %%
pbuoy = mesh.add_variable(nodeDofCount=mesh.dim)
projB = uw.utils.MeshVariable_Projection(pbuoy, fn=forceFn)

# %%
projB.solve()

# %%
fig2 = vis.Figure(figsize=(700,400), quality=2, rulers=True)
fig2.append( vis.objects.Surface(mesh, pbuoy[1], colourBar=True ) )
# fig1.append( vis.objects.Points(swarm, viscosityFn, fn_size=2.,colourBar = True, logScale=True  ) )
# fig1.append( vis.objects.Points(swarm, forceFn, fn_size=2.,colourBar = True, logScale=False  ) )
fig2.append( vis.objects.VectorArrows(mesh, 3e18*velocityField) )

# fig2.show()

# %%
# Visualise the result
vdotv  = fn.math.dot(velocityField,velocityField)
velmag = fn.math.sqrt(vdotv)

fig1 = vis.Figure(title="Uplift map - scaled viz", figsize=(700,400), quality=2, rulers=True)
# fig1.append( vis.objects.Surface(mesh, forceFn[1], colourBar=True ) )
# fig1.append( vis.objects.Points(swarm, viscosityFn, fn_size=2.,colourBar = True, logScale=True  ) )
fig1.append( vis.objects.Points(swarm, densityFn, absfn_size=2.,colourBar = True, logScale=False  ) )
fig1.append( vis.objects.VectorArrows(mesh, velocityField) )

# fig1.show()

# %%
# assign degrees of freedom (on each node) to be considered Dirichlet.
stokesBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                             indexSetsPerDof = (iWalls+bottomWall, jWalls) )

# %%
# setup solver
stokesPIC = uw.systems.Stokes( velocityField = velocityField, 
                               pressureField = pressureField,
                               conditions    = [stokesBC],
                               fn_viscosity  = viscosityFn, 
                               fn_bodyforce  = pbuoy )

#                                fn_one_on_lambda = lambdaFn )

solver = uw.systems.Solver( stokesPIC )
solver.set_inner_method("mumps")


# %%
# fig1.show()

# %%
# fig2.show()

# %%
# fields / variable to save
fieldDict = {'velocity':velocityField, 'pressure':pressureField}
swarmDict = {'material':materialVariable}

# %% code_folding=[]
# record output
outfile = open(output_path+'buildMount.txt', 'w+')
string = "steps, time, vrms, peak height"

if uw.mpi.rank==0:
    print(string)
    outfile.write( string+"\n")

# initialise loop
fn_y_minmax.reset()
fn_y_minmax.evaluate(mswarm)
h1    = fn_y_minmax.max_global()
h0    = h1
steps = 0
totalT = 0

# %% code_folding=[]
while steps<21:

    ## SOLVE ##
    # Get solution
    solver.solve()

    ## CHECKPOINT SAVE ##
    if steps > 0 and steps % 10 == 0:
        if (uw.mpi.rank == 0): print("CHECKPOINT SAVE")
        checkpoint( mesh, fieldDict, swarm, swarmDict, steps,
                    meshName='mesh', swarmName='swarm', time=dimensionalise(totalT, u.year).m, 
                    prefix='./notherun', enable_xdmf=True)
    
    ## ANALYTICS ##
    # update peak heigh
    fn_y_minmax.reset()
    fn_y_minmax.evaluate(mswarm)
    h1 = fn_y_minmax.max_global()

    diffH = h1-h0
    string = "{}, {:.3e}, {:.3e}, {:.3e}".format(steps,
                                     dimensionalise(totalT, u.kiloyear),
                                     dimensionalise(stokesPIC.velocity_rms(), u.cm/u.year),
                                     dimensionalise(diffH, u.metre) )
    if uw.mpi.rank == 0:
        print(string)
        outfile.write(string+"\n")
        
    fig1.save(output_path+"Uplift-"+str(steps)+".png")
    
    ## UPDATE IN TIME ##
    h0 = h1
    # Advect particles   
    advector.integrate(dt)  
    # Update buoyancy calculation
    projB.solve()
    msAdvector.integrate(dt)

    # population control
#     population_control.repopulate()

    totalT += dt
    steps += 1


outfile.close()

# %%
# 4, 5.000e+00 kiloyear, 7.639e-19 centimeter / year, 4.266e-17 meter

# %%
fig2.show()

# %%
# # for testing purposes
# dimensionalise=(diffH*dimensionalise(1,u.meter))
# if np.fabs(dimensionalise.magnitude-245.140) > 0.05*245.140:
#     raise RuntimeError("Height of passive tracer outside expected 5% tolerance")
