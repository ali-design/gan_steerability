import numpy as np

def batch_input(graph_inputs, s):
    '''
        graph_inputs: value from graph_input()
        s: slice of batch indices
    '''
    batched_input = {}
    for k,v in graph_inputs.items():
        if isinstance(v, np.ndarray):
            batched_input[k] = v[s]
        else:
            batched_input[k] = v
    return batched_input


def set_graph_kwargs(opt):
    graph_kwargs = dict(lr=opt.learning_rate, walk_type=opt.walk_type, loss=opt.loss)
    if opt.walk_type.startswith('NN'):
        if opt.nn.eps:
            graph_kwargs['eps'] = opt.nn.eps
        if opt.nn.num_steps:
            graph_kwargs['N_f'] = opt.nn.num_steps
    if opt.color.channel is not None and opt.transform.startswith("color"):
        graph_kwargs['channel'] = opt.color.channel
    if opt.model == 'stylegan':
        graph_kwargs['stylegan_opts'] = opt.stylegan
    if opt.model == 'pgan':
        graph_kwargs['pgan_opts'] = opt.pgan
    return graph_kwargs
