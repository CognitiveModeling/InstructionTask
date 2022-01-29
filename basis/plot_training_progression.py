import csv
import matplotlib.pyplot as plt

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

data_dict = {'epoch': [], 'trainloss': [], 'validloss': [], 'predloss': [], 'execloss': []}

with open('training_progression_0.001_50.csv','rt')as f:
  data = csv.reader(f)
  for row in data:
        data_dict['epoch'].append(row[0])
        data_dict['trainloss'].append(row[1])
        data_dict['validloss'].append(row[2])
        data_dict['predloss'].append(row[3])
        data_dict['execloss'].append(row[4])
epoch = [float(i) for i in data_dict['epoch']]
trainloss = [float(i) for i in data_dict['trainloss']]
validloss = [float(i) for i in data_dict['validloss']]
predloss = zero_to_nan([float(i) for i in data_dict['predloss']])
execloss = zero_to_nan([float(i) for i in data_dict['execloss']])

plt.plot(epoch, trainloss, color='blue', label='training loss')
plt.plot(epoch, validloss, color='red', label='validation loss')
plt.plot(epoch, predloss, 'o', color='green', label='prediction loss')
plt.plot(epoch, execloss, 'o', color='orange', label='execution loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
#plt.ylim(0, 0.148)
#plt.yscale('log')
plt.title('Loss progression')
plt.show()
