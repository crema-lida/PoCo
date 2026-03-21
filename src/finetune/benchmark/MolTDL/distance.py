import numpy as np


class mol_dis_sim:
    @staticmethod
    def Calculate_distance(Coor, arr_cutoff):
        num_atoms = len(Coor)
        all_atoms = []
        distance_matrix_real = np.ones((num_atoms, num_atoms), dtype=float)
        distance_matrix = np.ones((num_atoms, num_atoms), dtype=float)
        for i in range(num_atoms):
            all_atoms.append(Coor[i].split("\t")[0])
            for j in range(i + 1, num_atoms):
                x_i = float(Coor[i].split("\t")[1])
                y_i = float(Coor[i].split("\t")[2])
                z_i = float(Coor[i].split("\t")[3])

                x_j = float(Coor[j].split("\t")[1])
                y_j = float(Coor[j].split("\t")[2])
                z_j = float(Coor[j].split("\t")[3])

                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)

                if dis <= float(arr_cutoff[0]) or dis >= float(arr_cutoff[1]):
                    distance_matrix[i][j] = 0.0
                    distance_matrix[j][i] = 0.0
                    distance_matrix_real[i][j] = 0.0
                    distance_matrix_real[j][i] = 0.0
                else:
                    distance_matrix[i][j] = 1.0
                    distance_matrix[j][i] = 1.0
                    distance_matrix_real[i][j] = dis
                    distance_matrix_real[j][i] = dis

        return distance_matrix, distance_matrix_real, all_atoms
