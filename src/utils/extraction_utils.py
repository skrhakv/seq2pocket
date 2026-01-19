CLUSTER_DISTANCE = 0.5 # distance threshold to cut the hierarchical clustering tree, threshold follows the LBS comparison paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00923-z)
CLUSTERING_METHOD = 'average' # method to use for hierarchical clustering; see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

# This code was taken from the LIGYSIS repository (https://github.com/bartongroup/LIGYSIS/blob/running-arpeggio/ligysis.py); see LICENSE file

def get_intersect_rel_matrix(binding_ress):
    """
    Given a set of ligand binding residues, calculates a
    similarity matrix between all the different sets of ligand
    binding residues.
    """
    inters = {i: {} for i in range(len(binding_ress))}
    for i in range(len(binding_ress)):
        inters[i][i] = intersection_rel(binding_ress[i], binding_ress[i])
        for j in range(i+1, len(binding_ress)):
            inters[i][j] = intersection_rel(binding_ress[i], binding_ress[j])
            inters[j][i] = inters[i][j]
    return inters

def intersection_rel(l1, l2):
    """
    Calculates relative intersection.
    """
    len1 = len(l1)
    len2 = len(l2)
    I_max = min([len1, len2])
    I = len(list(set(l1).intersection(l2)))
    return I/I_max

def get_binding_site_clusters(binding_sites):
    """Get binding site clusters.

    Args:
        binding_sites (list): List of binding sites.
    """
    import scipy
    import pandas as pd
    
    irel_matrix = get_intersect_rel_matrix(binding_sites)
    irel_df = pd.DataFrame(irel_matrix)
    dist_df = 1 - irel_df # distance matrix in pd.Dataframe() format
    try:
        condensed_dist_mat = scipy.spatial.distance.squareform(dist_df) # condensed distance matrix to be used for clustering
        linkage = scipy.cluster.hierarchy.linkage(condensed_dist_mat, method=CLUSTERING_METHOD, optimal_ordering=True)
        return scipy.cluster.hierarchy.cut_tree(linkage, height=CLUSTER_DISTANCE)
    except ValueError:
        return None