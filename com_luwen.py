import networkx as nx
import community as community_louvain  # 需要 pip install python-louvain
from collections import defaultdict

def _invert_dict(d):
    """
    将 {node: community_id} 转为 {community_id: [nodes]}
    """
    result = defaultdict(list)
    for node, community in d.items():
        result[community].append(node)
    return result

def louvain_communities(G, k=None):
    """
    Louvain 社区发现算法
    Args:
        - G: 输入图
            + type: networkx.Graph
        - k: Louvain 不需要指定社区数，保留此参数是为了接口兼容
    Return:
        - List of communities, where each community is a list of vertex ID.
    """
    # Louvain 算法自动确定社区数
    partition = community_louvain.best_partition(G)
    return list(_invert_dict(partition).values())
