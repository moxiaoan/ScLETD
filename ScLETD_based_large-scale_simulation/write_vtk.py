nx=256#change
ny=256
nz=256
npoints=nx*ny*nz
directory = "./data/"
output = "./vtk/"
nstep=1100 
m = 0
for istep in range(800,nstep,200):
    # read origin dat file
    # eat1_.dat
    m = int(istep/100)
    if (m < 10):
    	fname = directory + "eta0_" + "00000" + str(m) + "_" + "000000" + ".dat"
    else:
    	fname = directory + "eta0_" + "0000" + str(m) + "_" + "000000" + ".dat"
    fp = open(fname, "r")
    data1 = fp.readlines()
    fp.close()
    # eta2_.dat
    #fname = directory + "eta2_" + str(istep) + ".dat"
    if (m < 10):
    	fname = directory + "eta1_" + "00000" + str(m) +"_" + "000000" + ".dat"
    else:
    	fname = directory + "eta1_" + "0000" + str(m) +"_" + "000000" + ".dat"
    fp = open(fname, "r")
    data2 = fp.readlines()
    fp.close()
    # eta3_.dat
    #fname = directory + "eta3_" + str(istep) + ".dat"
    if (m < 10):
    	fname = directory + "eta2_" + "00000" + str(m) + "_"+ "000000" + ".dat"
    else:
    	fname = directory + "eta2_" + "0000" + str(m) + "_"+ "000000" + ".dat"
    fp = open(fname, "r")
    data3 = fp.readlines()
    fp.close()
    # eta4_.dat
    #fname = directory + "eta4_" + str(istep) + ".dat"
    if (m < 10):
    	fname = directory + "eta3_" + "00000" + str(m) +"_"+ "000000" + ".dat"
    else:
    	fname = directory + "eta3_" + "0000" + str(m) +"_"+ "000000" + ".dat"
    fp = open(fname, "r")
    data4 = fp.readlines()
    fp.close()
    if (m < 10):
    	fname = directory + "eta4_" + "00000" + str(m) +"_"+ "000000" + ".dat"
    else:
    	fname = directory + "eta4_" + "0000" + str(m) +"_"+ "000000" + ".dat"
    fp = open(fname, "r")
    data5 = fp.readlines()
    fp.close()
    if (m < 10):
    	fname = directory + "eta5_" + "00000" + str(m) +"_"+ "000000" + ".dat"
    else:
    	fname = directory + "eta5_" + "0000" + str(m) +"_"+ "000000" + ".dat"
    fp = open(fname, "r")
    data6 = fp.readlines()
    fp.close()
    # etasum_.dat
    #fname = directory + "etasum_" + str(istep) + ".dat"
    #fp = open(fname, "r")
    #datasum = fp.readlines()
    #fp.close()

    # write to vtk file
    fname = output + "time_alpha_" + str(istep) + ".vtk"
    fp = open(fname, "w")

    # header of vtk file
    fp.write('# vtk DataFile Version 2.0\n')
    fp.write('time_{0:d}.vtk\n'.format(istep))
    fp.write('ASCII\n')
    fp.write('DATASET STRUCTURED_GRID\n')

    # coords of grid points
    fp.write('DIMENSIONS {0:>5d}  {1:>5d}  {2:>5d}\n'.format(nx,ny,nz))#change
    fp.write('POINTS {0:>7d} float\n'.format(npoints))#change
    #for i in range(0,nx):
    #    for j in range(0,ny):
    #        for k in range(0,nz):
    #            fp.write('{0:>14.6e}   {1:>14.6e}   {2:>14.6e}\n'.format(1.0*i,1.0*j,1.0*k))
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                fp.write('{0:>14.6e}   {1:>14.6e}   {2:>14.6e}\n'.format(1.0*i,1.0*j,1.0*k))

    # write grid point values:
    fp.write('POINT_DATA {0:>5d}\n'.format(npoints))#change

    fp.write('SCALARS eta0  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data1:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))

    fp.write('SCALARS eta1  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data2:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))

    fp.write('SCALARS eta2  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data3:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))
    
    fp.write('SCALARS eta3  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data4:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))

    fp.write('SCALARS eta4  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data5:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))

    fp.write('SCALARS eta5  float  1\n')
    fp.write('LOOKUP_TABLE default\n')
    for s in data6:
        v = float(s.strip())
        fp.writelines("{0:>14.6e}\n".format(v))
    #fp.write('SCALARS etasum  float  1\n')
    #fp.write('LOOKUP_TABLE default\n')
    #for s in datasum:
    #    v = float(s.strip())
    #    fp.writelines("{0:>14.6e}\n".format(v))

    fp.close()





