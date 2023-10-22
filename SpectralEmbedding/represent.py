
from data.data_functions import *
import scipy.stats

def get_eigenvec(graph, params):
    """
    Calcaulate eigenvalues and eigenvectors of the graph

    """
    rep_type = params["rep_type"]
    matrix_type = params["matrix_type"]
    if matrix_type == "adjacency":
        matrix = nx.adjacency_matrix(graph).todense()
    elif matrix_type == "laplacian":
        matrix = nx.laplacian_matrix(graph).todense()
    elif matrix_type == "laplacian_norm":
        matrix = nx.normalized_laplacian_matrix(graph).todense()
    elif "attributes" in matrix_type:
        matrix = get_attributes_matrix(graph, matrix_type)
        matrix = matrix[:, np.where(np.std(matrix, axis=0) > 0.000001)[0]]
        if rep_type == "matrix_padded":
            # drop columns with all same values
            temp = np.zeros((max(matrix.shape), max(matrix.shape)))
            temp[:matrix.shape[0], :matrix.shape[1]] = matrix
            matrix = temp
    else:
        print("Invalid matrix type")
        exit(1)
    if "svd" in rep_type:
        u, s, v = np.linalg.svd(matrix)
        eigenvalues = s
        eigenvectors = u
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    return eigenvalues_process(degrees, eigenvalues, eigenvectors, params)


def eigenvalues_process(degrees, eigenvalues, eigenvectors, params):
    n_vertices_need = params["n_vertices"]
    n_eigenvalues_need = params["n_eigenvalues"]
    absolute_value = params["absolute_value"]
    eigenvalues_ind = np.argsort(eigenvalues)

    degrees_ind = [x[0] for x in degrees]

    eigenvectors_to_take = eigenvectors[degrees_ind[:n_vertices_need], :]
    eigenvectors_to_take = eigenvectors_to_take[:, eigenvalues_ind[:n_eigenvalues_need]]

    eigenvalues_to_take = eigenvalues[eigenvalues_ind[:n_eigenvalues_need]]
    # padding
    n_nodes = eigenvectors_to_take.shape[0]
    n_eigenvalues = eigenvectors_to_take.shape[1]
    if n_nodes < n_vertices_need:
        eigenvectors_to_take = np.pad(eigenvectors_to_take, ((0, n_vertices_need - n_nodes), (0, 0)))

    if n_eigenvalues < n_eigenvalues_need:
        eigenvectors_to_take = np.pad(eigenvectors_to_take, ((0, 0), (0, n_eigenvalues_need - n_eigenvalues)))
        eigenvalues_to_take = np.pad(eigenvalues_to_take, (0, n_eigenvalues_need - n_eigenvalues))

    eigenvalues_to_take = np.real(eigenvalues_to_take)
    eigenvectors_to_take = np.real(eigenvectors_to_take)
    if absolute_value:
        eigenvectors_to_take = np.abs(eigenvectors_to_take)

    return eigenvectors_to_take, eigenvalues_to_take

def get_eigenvectors_moments(graph, params):
    """ Get a graph and return the eigenvalues and the eigenvectors momemts """
    eigenvectors, eigenvalues = get_eigenvec(graph, params)
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)
    # calculate moments for each column
    eigenvectors_moments = np.apply_along_axis(lambda x: calc_moments(x, n_moments=params["n_moments"]),
                                               0, eigenvectors)
    return eigenvectors_moments, eigenvalues


def calc_moments(vec, n_moments=10):
    """ Calculate the first n_moments of the vector. For odd moments we calculate the moments of the absolute value """
    return np.array([
        scipy.stats.moment(vec, i) if i % 2 ==0 else scipy.stats.moment(abs(vec), i) for i in range(n_moments)
        ])

def get_mean_attributes(graph, attribute_type):
    """ Get a graph and return the mean of the nodes attributes"""
    attribute = get_attributes_matrix(graph, attribute_type)
    return np.mean(attribute, axis=0)


if __name__ == '__main__':
 pass