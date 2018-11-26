from glob import glob
import numpy as np
import math

files = glob('deploy/*/*/*_image.jpg')
for idx in range(len(files)):
    snapshot = files[idx]
    print(snapshot)

    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])

    # user input: number of segments in x, y, z - axis
    xdim = 200; ydim = 20; zdim = 200;

    # user input: filt the cloud coordinates in the specified range
    globalxrange = [-50, 50]
    globalyrange = [-10, 5]
    globalzrange = [0, 80]

    xintv = (globalxrange[1] - globalxrange[0]) / xdim
    yintv = (globalyrange[1] - globalyrange[0]) / ydim
    zintv = (globalzrange[1] - globalzrange[0]) / zdim
    globalgrid = np.zeros((xdim,ydim,zdim), dtype=int)
    for cloudi in range(xyz.shape[1]):
        modx = math.floor((xyz[0, cloudi] - globalxrange[0]) / xintv)
        mody = math.floor((xyz[1, cloudi] - globalyrange[0]) / yintv)
        modz = math.floor((xyz[2, cloudi] - globalzrange[0]) / zintv)
        if (modx >= 0 and modx <= xdim-1 and mody >= 0 and mody <= ydim-1 \
            and modz >= 0 and modz <= zdim-1):
            if(globalgrid[modx, mody, modz] < 255):
                globalgrid[modx, mody, modz] += 1

    globalgrid = globalgrid.reshape([1, -1])

    globalgrid2 = globalgrid.tolist()[0]
    newfile = open(snapshot.replace('_image.jpg', '_grid.bin'), 'wb')
    newfile.write(bytearray(globalgrid2))
    
    # ================================================================
    # how to read from *_grid.bin
    # points is a list containing number of cloud points in every grid
    # ================================================================
    # points = np.fromfile(snapshot.replace('_image.jpg', '_grid.bin'), dtype=np.int8)



