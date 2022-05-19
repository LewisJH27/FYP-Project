import os

f = open('snaplist2.txt', 'r')
for i in range(0, len(list(f))):
    #print(i)
    os.system("mpirun -np 4 python core/mainMEGA.py params/mega-param_mpitest_lewisOC.yml "+str(i))

# os.system("mpirun -np 4 python core/mainMEGA.py params/mega-param_mpitest_lewisOC.yml 98")