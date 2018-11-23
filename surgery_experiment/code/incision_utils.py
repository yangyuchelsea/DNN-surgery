import numpy as np
import sys
# from keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def manual_incision_generation(model,ring_left,ring_right):
    print('generate incision...')
    singe_incision = []
    for i in range(1, (len(model.layers) - 2)):
        w = [(i, 1)]
        singe_incision.append(w)


    if len(ring_left) == len(ring_right):
        ring_incision = [[(k, 1), (m, 1)] for i in range(len(ring_left)) for k in ring_left[i] for m in ring_right[i]]
    else:

        print('wrong ring information', len(ring_left), len(ring_right))
        sys.exit()

    incision = singe_incision + ring_incision
    if len(singe_incision) + len(ring_incision) == len(candidate_incision):
        print('incision prepared')
    else:
        sys.exit()
    return incision

def incision_generation(model):
    dic_id = {}
    edge_id = []
    for idx, layer in enumerate(model.layers):
        layer_id = str(id(layer))
        dic_id[layer_id] = idx
        for node in layer._inbound_nodes:
            for inbound_layer in node.inbound_layers:
                
                edge_id.append((str(id(inbound_layer)), layer_id))
            
    edge = [(dic_id[i[0]],dic_id[i[1]]) for i in edge_id]
    ##generate adjacency_matrix
    adjacency_matrix = np.zeros(shape=(len(model.layers),len(model.layers)))
    for each_edge in edge:
        adjacency_matrix[each_edge[0],each_edge[1]] =1
        adjacency_matrix[each_edge[1],each_edge[0]] =-1 
    forward_adjacency_matrix = adjacency_matrix.clip(min=0)
    ##get all start and end of rings
    ring_start = np.where(forward_adjacency_matrix.sum(axis=1)  >1)[0].tolist()
    end_of_ring = np.where(adjacency_matrix.clip(max=0).sum(axis=1) <-1)[0].tolist()
    if len(ring_start) != len(end_of_ring):
        print('incorret ring start and end')
    ##genrate wound
    edge_list = []
    idx = 1
    residule_no = 0
    inception_no = 0
    while idx < len(model.layers):
        
        if idx in ring_start: 
            if end_of_ring[ring_start.index(idx)] in np.where(adjacency_matrix[idx,:] == 1)[0]:
                residule_no += 1
                for j in range(idx,end_of_ring[ring_start.index(idx)]+1):
                    edge_list.append([j])
            else:
                inception_no += 1 
                if(sum(adjacency_matrix.clip(min=0)[idx,:]) != 2):
                    print('generate wrong rings')
                    #sys.exit()
                left = [np.where(adjacency_matrix[idx,:] == 1)[0][0]]
                right = [np.where(adjacency_matrix[idx,:] == 1)[0][1]]
                j = idx + 1
                while j < end_of_ring[ring_start.index(idx)]:
                    if j in left:
                        left.append(np.where(adjacency_matrix[j,:] == 1)[0][0])
                    elif j in right:
                        right.append(np.where(adjacency_matrix[j,:] == 1)[0][0])
                    j= j+1
                for m in range(idx,end_of_ring[ring_start.index(idx)]):
                    edge_list.append([m])
                if left[-1] == right[-1] == end_of_ring[ring_start.index(idx)]:
                    edge_list.append([left[:len(left)-1],right[:len(right)-1]])
                else:
                    print('generate wrong rings')
                   # sys.exit()
                edge_list.append([end_of_ring[ring_start.index(idx)]])
            idx = end_of_ring[ring_start.index(idx)] +1
        else:
            edge_list.append([idx])
            idx += 1
    wound = []
    for edge in edge_list:
        if len(edge) == 1:
            wound.append([edge[0]])
        elif len(edge) == 2:
            for m in edge[0]:
                for k in edge[1]:
                     wound.append([m,k])

    incision = []
    for edge in edge_list:
        if len(edge) == 1:
            incision.append([(edge[0], 1)])
        elif len(edge) == 2:
            for m in edge[0]:
                for k in edge[1]:
                     incision.append([(m, 1), (k, 1)])
    return residule_no,inception_no,wound,incision



def wound2incision(wound):
    incision = []
    for edge in wound:
        if len(edge) == 1:
            incision.append([(edge[0], 1)])
        elif len(edge) == 2:
            incision.append([(edge[0], 1), (edge[1], 1)])

    return incision


def generate_cut(model,incision):
    layer_name_idx = {}
    for idx, layer in enumerate(model.layers):
        if idx in [i[0] for i in incision]:
            layer_name_idx[idx]= layer.name

    true_output_shape= []
    for i in range(len(incision)):
        true_output_shape_of_layer = tuple()
        for j in range(1,len(model.layers[incision[i][0]].output.shape)):
            true_output_shape_of_layer = true_output_shape_of_layer + (model.layers[incision[i][0]].output.shape[j].value,)
        true_output_shape.append(true_output_shape_of_layer)

    cut = []
    for i in incision:
        if i[1] == 0:
            cut.append(model.layers[i[0]].input)

        elif i[1] == 1:
            cut.append(model.layers[i[0]].output)
        else:
            print('incorrect wound, check cut layers')
    return cut,true_output_shape,layer_name_idx


def feasible_wound_generate(model,wound,incision,x):

    feasible_incision = []
    feasible_wound = []
    for i in range(len(incision)-1):
        cut, true_output_shape, layer_name_idx = generate_cut(model,incision[i])
        transplant = K.function([model.input],cut)
        transplant_feature = transplant([x])
        try:
            donor = K.function(cut,[model.output])
            predict = donor(transplant_feature)
            result = decode_predictions(predict[0], top=1)[0]
            #print(result)
            feasible_wound.append(wound[i])
            feasible_incision.append(incision[i])
        except:
            pass
    return feasible_wound,feasible_incision






