import pydot
import numpy as np
import sys


def plot_model(idx_flag, model, to_file, show_shapes, show_layer_names):
    if idx_flag:
        plot_model_with_idx(model, to_file, show_shapes, show_layer_names)
    else:
        plot_model_without_idx(model, to_file, show_shapes, show_layer_names)

    
def plot_model_with_idx(model,
               to_file = True,
               show_shapes = True,
               show_layer_names = True,
               rankdir='TB'):
    
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    
    for idx, layer in enumerate(model.layers):
        layer_id = str(id(layer))
        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        if show_shapes:
            
            label = '%s|%s\n|{input:|output:}|{{%s}|{%s}}' % (idx,label,str(layer.input_shape),str(layer.output_shape))
        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)
        
    for layer in model.layers:
        layer_id = str(id(layer))
        for node in layer._inbound_nodes:
            for inbound_layer in node.inbound_layers:
                  dot.add_edge(pydot.Edge(str(id(inbound_layer)), layer_id))
    dot.write_png(to_file)
