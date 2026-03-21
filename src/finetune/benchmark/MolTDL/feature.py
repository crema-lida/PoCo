import numpy as np


class mol_feature:
    @staticmethod
    def Calculate_feature(dis_mat, all_atoms):
        feat_mat = np.zeros((len(all_atoms), 62), dtype=float)
        for a in range(len(dis_mat[0, :])):
            for b in range(len(dis_mat[:, 0])):
                if dis_mat[a][b] == 0:
                    continue
                if all_atoms[b] == "C":
                    if dis_mat[a][b] >= 10:
                        feat_mat[a, 18] += 1
                    else:
                        feat_mat[a, int((dis_mat[a, b] - 1) * 2)] += 1
                elif all_atoms[b] == "H":
                    if dis_mat[a][b] >= 10:
                        feat_mat[a, 37] += 1
                    else:
                        feat_mat[a, 19 + int((dis_mat[a, b] - 1) * 2)] += 1
                elif all_atoms[b] == "O":
                    if dis_mat[a][b] < 2.5:
                        feat_mat[a, 38] += 1
                    elif dis_mat[a][b] < 5:
                        feat_mat[a, 39] += 1
                    elif dis_mat[a][b] < 7.5:
                        feat_mat[a, 40] += 1
                    else:
                        feat_mat[a, 41] += 1
                elif all_atoms[b] == "N":
                    if dis_mat[a][b] <= 2.5:
                        feat_mat[a, 42] += 1
                    elif dis_mat[a][b] < 5:
                        feat_mat[a, 43] += 1
                    elif dis_mat[a][b] < 7.5:
                        feat_mat[a, 44] += 1
                    else:
                        feat_mat[a, 45] += 1
                elif all_atoms[b] == "P":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 46] += 1
                    else:
                        feat_mat[a, 47] += 1
                elif all_atoms[b] == "Cl" or all_atoms[a] == "CL":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 48] += 1
                    else:
                        feat_mat[a, 49] += 1
                elif all_atoms[b] == "F":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 50] += 1
                    else:
                        feat_mat[a, 51] += 1
                elif all_atoms[b] == "Br":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 52] += 1
                    else:
                        feat_mat[a, 53] += 1
                elif all_atoms[b] == "S":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 54] += 1
                    else:
                        feat_mat[a, 55] += 1
                elif all_atoms[b] == "Si":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 56] += 1
                    else:
                        feat_mat[a, 57] += 1
                elif all_atoms[b] == "I":
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 58] += 1
                    else:
                        feat_mat[a, 59] += 1
                else:
                    if dis_mat[a][b] < 5:
                        feat_mat[a, 60] += 1
                    else:
                        feat_mat[a, 61] += 1
        return feat_mat
