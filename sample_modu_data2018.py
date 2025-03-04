import numpy as np
import h5py

##############################全局参数#######################################
# f = h5py.File('C:/Users/Winner/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
f = h5py.File('GOLD_XYZ_OSC.0001_1024.hdf5','r')
dir_path = './data/RML2018_modu_10_26'
modu_snr_size = 100
modu_snr_size = 1200
############################################################################

for modu in range(24):  # 24种调制方式
	X_list = []
	Y_list = []
	Z_list = []
	print('part ',modu)
	start_modu = modu*106496  # 每种调制方式有106496条数据
	for snr in range(10, 26):  # 每种信噪比下有4096条数据
		if snr < 10:
			start_snr = start_modu + snr * 4096
			idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
			X = f['X'][start_snr:start_snr + 4096][idx_list]
			# X = X[:,0:768,:]
			X_list.append(X)
			Y_list.append(f['Y'][start_snr:start_snr + 4096][idx_list])
			Z_list.append(f['Z'][start_snr:start_snr + 4096][idx_list])
		else:
			start_snr = start_modu + snr * 4096
			idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
			X = f['X'][start_snr:start_snr + 4096][idx_list]
			# X = X[:,0:768,:]
			X_list.append(X)
			Y_list.append(f['Y'][start_snr:start_snr + 4096][idx_list])
			Z_list.append(f['Z'][start_snr:start_snr + 4096][idx_list])

	filename = dir_path + '/part' + str(modu) + '.h5'
	fw = h5py.File(filename,'w')
	fw['X'] = np.vstack(X_list)
	fw['Y'] = np.vstack(Y_list)
	fw['Z'] = np.vstack(Z_list)
	print('X shape:',fw['X'].shape)
	print('Y shape:',fw['Y'].shape)
	print('Z shape:',fw['Z'].shape)
	fw.close()
f.close()