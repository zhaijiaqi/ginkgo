import os
import sys
import csv

filename = "valid_matrix_set.csv"

total = sum(1 for line in open(filename))
print(total)

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for i in range(1, total):
        cur_row = next(csv_reader)
        matrix_group = "MM/" + cur_row[1]
        matrix_name = cur_row[2]
        matrix_path_in_matrix_folder = os.path.join(os.path.expanduser("~/data/matrix"), matrix_name + ".mtx")
        # 先判断~/data/matrix文件夹下是否已有该矩阵
        if os.path.exists(matrix_path_in_matrix_folder):
            print(f"--- Matrix {matrix_name} already exists in ~/data/matrix folder ---")
            continue
        # 再判断MM下的结构是否存在（维持原有逻辑，防止再次下载）
        if os.path.exists(matrix_group + "/" + matrix_name + "/" + matrix_name + ".mtx") == False:
            matrix_url = "http://sparse-files.engr.tamu.edu/MM/" + cur_row[1] + "/" + cur_row[2] + ".tar.gz"
            os.system("wget " + matrix_url)
            os.system("tar -zxvf " + matrix_name + ".tar.gz " + "-C " + "./")
            os.system("mv " + matrix_name+"/"+ matrix_name+".mtx ~/data/matrix")
            os.system("rm -rf " + matrix_name + ".tar.gz")
            os.system("rm -rf " + matrix_name)
