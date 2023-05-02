import numpy as np

def get_P(cab_f):
    for line in open(cab_f):
        if 'P_rect_02' in line or 'P2:' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix = cam_P.reshape((3,4))
            return return_matrix

    print("Error: File not found at", cab_f)

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line or 'R_rect_02:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))
            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0
            return R0_rect

    print("Error: File not found at", cab_f)

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line or 'T_02:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (1, 3))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,3] = Tr
            print(Tr_to_velo)
            return Tr_to_velo

    print("Error: File not found at", cab_f)