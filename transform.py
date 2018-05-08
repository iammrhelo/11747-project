import torch 
import numpy
import pickle

for k in range(2):
    print(k)
    data = pickle.load( open( "valid_extracted_data_" + str(k+1) + ".p", "rb" ) )
    data_ret = []
    for each in data:
        context = [i.data.cpu().numpy() for i in each[0]]
        entities = [i.data.cpu().numpy() for i in each[1]]
        (h, c) = (each[2][0].data.cpu().numpy(), each[2][1].data.cpu().numpy())
        target = each[3]

        data_ret.append((context, entities, (h, c), target))

    pickle.dump( data_ret, open( "numpy_" + "valid_extracted_data_" + str(k+1) + ".p", "wb" ) )
