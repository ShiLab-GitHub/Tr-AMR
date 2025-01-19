import numpy as np
import h5py
import pickle

##############################全局参数#######################################
# f = h5py.File('C:/Users/Winner/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
f = h5py.File('GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
dir_path = './data/RML2018_snr'
modu_snr_size = 1200  # 每个信噪比下面会有4096个数据，但是只使用1200个
############################################################################

for snr in range(26):  # 26种信噪比
    X_list = []
    Y_list = []
    Z_list = []
    print('part ', snr)
    start_snr = snr * 4096  # 每种调制方式有106496条数据每种信噪比下有4096条数据
    for modu in range(24):
        start_modu = start_snr + modu * 106496
        idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
        X = f['X'][start_modu:start_modu + 4096][idx_list]
        X_list.append(X)
        Y_list.append(f['Y'][start_modu:start_modu + 4096][idx_list])
        Z_list.append(f['Z'][start_modu:start_modu + 4096][idx_list])

    filename = dir_path + '/part' + str(snr) + '.h5'
    fw = h5py.File(filename, 'w')
    fw['X'] = np.vstack(X_list)
    fw['Y'] = np.vstack(Y_list)
    fw['Z'] = np.vstack(Z_list)
    print('X shape:', fw['X'].shape)
    print('Y shape:', fw['Y'].shape)
    print('Z shape:', fw['Z'].shape)
    fw.close()
f.close()