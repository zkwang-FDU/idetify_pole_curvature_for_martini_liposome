curve_file_1="data_perimeter.txt"
curve_file_2="data_perimeter_2.txt"

with open(curve_file_1,'r') as f1:
    line1 = f1.readlines()
    data1 = []
    for line in line1:
        data1.append(line.split()[1])
        data1_=data1[1:]

with open(curve_file_2,'r') as f2:
    line2 = f2.readlines()
    data2 = []
    for line in line2:
        data2.append(line.split()[1])
        data2_=data2[1:]

compare = []
for i in range(len(data1)-1):
    if float(data1_[i]) != 0 and float(data2_[i])!= 0 :
        compare.append(data1_[i])
        compare.append((float(data1_[i])-float(data2_[i]))/float(data1_[i]))

print(compare)
