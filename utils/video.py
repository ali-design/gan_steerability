import skvideo.io
import numpy as np


rate = '30'
inputdict = {
    '-r': rate
}
outputdict = {
    '-pix_fmt': 'yuv420p',
    '-r': rate
}

def make_video(graph, graph_inputs, filename, num_frames=100,
              max_alpha=None, min_alpha=None, channel=None):

    if max_alpha is not None and min_alpha is not None:
        alphas = np.linspace(min_alpha, max_alpha, num_frames)
    else:
        alphas = graph.vis_alphas(num_frames)

    zs_batch = graph_inputs[graph.z]
    if graph.walk_type.startswith('NN') and graph.channel is not None:
        # NN walk was trained on a specific channel, use that channel
        # for walk
        channel = graph.channel
        print("Using channel: {}".format(graph.channel))

    writer = skvideo.io.FFmpegWriter(filename,
                                     inputdict,
                                     outputdict)
    for a in alphas:
        a_graph = graph.scale_test_alpha_for_graph(
            a, zs_batch, channel=channel, contrast=True)
        best_im_out = graph.apply_alpha(graph_inputs, a_graph)
        writer.writeFrame(graph.clip_ims(best_im_out))
    writer.close()
