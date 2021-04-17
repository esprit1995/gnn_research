import plotly.express as px
import torch
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA


def draw_embeddings(embeddings: torch.tensor,
                    cluster_labels: torch.tensor,
                    node_type_mask: torch.tensor,
                    path_to_save: str = '',
                    name_to_save: str = 'cool_plotly_vis.html'):
    """
    produce an interactive plotly visualization: apply PCA to reduce the dim of the given
    embeddings to 3; color by cluster label, shape by type mask
    :param embeddings: node embeddings in their original form (torch.tensor)
    :param cluster_labels: cluster labels of the embeddings (torch.tensor)
    :param node_type_mask: node type mask tensor (torch.tensor)
    :param path_to_save: where to save the html visualization to
    :param name_to_save: name under which to save the html visualization
    :return: None
    """
    pca = PCA(n_components=3)
    compressed_embs = pca.fit_transform(embeddings.detach().numpy())
    df = pd.DataFrame(data=compressed_embs, columns=['pc1', 'pc2', 'pc3'])
    df['type'] = node_type_mask.numpy()
    df['cluster'] = cluster_labels
    fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3',
                        color='cluster', symbol='type')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_html(os.path.join(path_to_save, name_to_save))
    return
