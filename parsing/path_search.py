import networkx as nx


def shortest_paths(graph, source, target, cutoff=None):
    return nx.all_shortest_paths(graph, source, target)


def shortest_paths_plus_threshold(graph, source, target, cutoff=1):
    shortest_path_length = nx.shortest_path_length(graph, source, target)
    return nx.all_simple_paths(graph, source, target, cutoff=shortest_path_length + cutoff)


def all_paths(graph, source, target, cutoff=8):
    return nx.all_simple_paths(graph, source, target, cutoff=cutoff)
