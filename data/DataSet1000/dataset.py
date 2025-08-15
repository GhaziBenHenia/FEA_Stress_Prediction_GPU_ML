# import modules ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pyvista as pv
from os.path import exists

pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'
pv.global_theme.colorbar_orientation = 'vertical'  

stress_file_exists = exists('data/stress0.vtu')
overwrite_stress_file = 0

# macro
print("----------------------------------------------")
print("--- Von Mises Stress without heterogeneity ---")
print("----------------------------------------------")

#reader = pv.get_reader("data/vm0.pvd")
#vm0 = reader.read()[0]
#vm0.plot(scalars='sigma_vm',show_edges=True)

plotter = pv.Plotter()    
reader = pv.get_reader("data/vm0.pvd")
vm0 = reader.read()[0]
plotter.add_mesh(vm0, scalars="sigma_vm",show_edges=True)
plotter.store_image = True
plotter.show()
plotter.screenshot('vm0.png')

print("------------------------------------")
print("--- Stress without heterogeneity ---")
print("------------------------------------")



if stress_file_exists==1 and overwrite_stress_file==0:

    print("Read stress file")
    reader = pv.get_reader("data/stress0.vtu")
    stress0 = reader.read()

else:   

    print("Build stress file with 6 componenets from two separate files (FE code could only outputs 3 components at a time)")
    print("Read first three components of stress tensor in voigt notations")

    # macro stress first three components (bulk)
    reader = pv.get_reader("data/stress_bulk0.pvd")
    stress_bulk0 = reader.read()[0]  # MultiBlock mesh with only 1 block
    print("stress_bulk0",stress_bulk0)
    print("stress_bulk0.array_names",stress_bulk0.array_names)
    stress_bulk0.plot(scalars='sigma_bulk', component=0)
    stress_bulk0.plot(scalars='sigma_bulk', component=1)
    stress_bulk0.plot(scalars='sigma_bulk', component=2)
    stress_bulk0_array = pv.get_array(stress_bulk0,'sigma_bulk')
    print("stress_bulk0_array",stress_bulk0_array)

    #reader = pv.get_reader("stress_bulk0.pvd")
    #stress_bulk0 = reader.read()[0]
    #stress_bulk0.plot(scalars='stress_bulk', component=2)

    print("Read last three components of stress tensor in voigt notations")

    # macro stress last three components (shear)
    reader = pv.get_reader("data/stress_shear0.pvd")
    stress_shear0 = reader.read()[0]
    stress_shear0.plot(scalars='sigma_shear', component=0)
    stress_shear0.plot(scalars='sigma_shear', component=1)
    stress_shear0.plot(scalars='sigma_shear', component=2)
    stress_shear0_array = pv.get_array(stress_shear0,'sigma_shear')

    print("Concatenate first and last three components into a nodel 6-dimensional stress vector")

    # concatenate all stress components
    print("stress_bulk0_array.shape",stress_bulk0_array.shape)
    print("stress_shear0_array.shape",stress_shear0_array.shape)
    stress0_array = np.concatenate((stress_bulk0_array,stress_shear0_array), axis=1,)
    print("stress0_array.shape",stress0_array.shape)

    # full stress
    #stress0 = pv.UnstructuredGrid()
    #stress0.points = stress_bulk0.points
    #stress0.cells = stress_bulk0.cells
    stress0 = stress_bulk0
    stress0.clear_data()
    #stress0.add_field_data(stress0_array, 'stress')
    stress0['stress'] = stress0_array
    print("stress0.array_names",stress0.array_names)
    stress0.plot(scalars='stress', component=0)

    stress0.save("data/stress0.vtu")

plotter = pv.Plotter()    
plotter.add_mesh(stress0, scalars='stress', component=5)
plotter.store_image = True
plotter.show()
plotter.screenshot('stress0_12.png')


# realisations
print("-------------------------------------------")
print("----------- random realisations -----------")
print("-------------------------------------------")

print("realisation 1: Elasticity modulus")

# random field of Young's modulus
plotter = pv.Plotter()    
reader = pv.get_reader("data/rand_E1.pvd")
E = reader.read()[0]
plotter.add_mesh(E, scalars="E")
plotter.store_image = True
plotter.show()
plotter.screenshot('E1.png')

print("realisation 1: Von Mises stress")

# Von Mises stress
plotter = pv.Plotter()    
reader = pv.get_reader("data/vm1.pvd")
vm = reader.read()[0]
plotter.add_mesh(vm, scalars="sigma_vm")
plotter.store_image = True
plotter.show()
plotter.screenshot('vm1.png')

print("realisation 2: Elasticity modulus")

# random field of Young's modulus
plotter = pv.Plotter()    
reader = pv.get_reader("data/rand_E2.pvd")
E = reader.read()[0]
plotter.add_mesh(E, scalars="E")
plotter.store_image = True
plotter.show()
plotter.screenshot('E2.png')

print("realisation 2: Von Mises stress")

# Von Mises stress
plotter = pv.Plotter()    
reader = pv.get_reader("data/vm2.pvd")
vm = reader.read()[0]
plotter.add_mesh(vm, scalars="sigma_vm")
plotter.store_image = True
plotter.show()
plotter.screenshot('vm2.png')

print("realisation 100: Elasticity modulus")

# random field of Young's modulus
plotter = pv.Plotter()    
reader = pv.get_reader("data/rand_E100.pvd")
E = reader.read()[0]
plotter.add_mesh(E, scalars="E")
plotter.store_image = True
plotter.show()
plotter.screenshot('E100.png')

print("realisation 100: Von Mises stress")

# Von Mises stress
plotter = pv.Plotter()    
reader = pv.get_reader("data/vm100.pvd")
vm = reader.read()[0]
plotter.add_mesh(vm, scalars="sigma_vm")
plotter.store_image = True
plotter.show()
plotter.screenshot('vm100.png')

print("Done reading data. Good day")

# random field of Young's modulus
plotter = pv.Plotter()    
reader = pv.get_reader("data/rand_E1000.pvd")
E = reader.read()[0]
plotter.add_mesh(E, scalars="E")
plotter.store_image = True
plotter.show()
plotter.screenshot('E1000.png')

print("realisation 1000: Von Mises stress")

# Von Mises stress
plotter = pv.Plotter()    
reader = pv.get_reader("data/vm1000.pvd")
vm = reader.read()[0]
plotter.add_mesh(vm, scalars="sigma_vm")
plotter.store_image = True
plotter.show()
plotter.screenshot('vm1000.png')

print("Done reading data. Good day")