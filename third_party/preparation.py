import numpy as np
import os.path
filesave = input('Output file:\n    ')

Num_Para = 67


is_first = True
arrdata = []
arrlabel = []
dataset = []
labelset = []
for fileline in open('datalist.txt'):
    if(len(fileline)==0):
        break
    for line in open('../dataout/' + fileline + '.txt'):
        if line.split(',')[0] == '0' or line.split(',')[0] == '5':
            continue
        if line.split(',')[0] =='1':
            arrlabel = [1,0,0,0]
        if line.split(',')[0] == '2':
            arrlabel = [0, 1, 0, 0]
        if line.split(',')[0] =='3':
            arrlabel = [0,0,1,0]
        if line.split(',')[0] =='4':
            arrlabel = [0,0,0,1]
        for i in range(1,Num_Para*2+1):
            arrdata.append(int(line.split(',')[i]))
        if is_first:
            dataset = arrdata
            labelset = arrlabel
            is_first = False
        else:
            dataset = np.vstack((dataset,arrdata))
            labelset = np.vstack((labelset,arrlabel))
        arrdata.clear()

if os.path.isfile('./processed/'+filesave+'_data.npy'):
    ytrain = np.load('./processed/'+filesave+'_data.npy')
    ylabel = np.load('./processed/'+filesave+'_label.npy')
    ytrain = np.vstack((ytrain,dataset))
    ylabel = np.vstack((ylabel,labelset))
    np.save(r'./processed/'+filesave+'_data.npy',ytrain)
    np.save(r'./processed/' + filesave + '_label.npy', ylabel)
    print(len(ytrain))
else:
    np.save(r'./processed/'+filesave+'_data.npy',dataset)
    np.save(r'./processed/'+filesave+'_label.npy',labelset)
    print(len(dataset))
print('Finished')