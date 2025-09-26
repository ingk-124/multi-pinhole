# distutils: language = c++
# distutils: boundscheck = False
# distutils: wraparound = False
# distutils: cdivision = True

import numpy
cimport numpy
ctypedef numpy.float64_t DOUBLE_t


def say_hello_to(name):
    print(f"Hello {name}\!")

# def check_intersection(cnp.ndarray[DTYPE_t, ndim=2] triangle,
#                        cnp.ndarray[DTYPE_t, ndim=1] start,
#                        cnp.ndarray[DTYPE_t, ndim=2] end_points) -> np.ndarray:
#     cdef int _n = end_points.shape[0]
#     cdef cnp.ndarray[DTYPE_t, ndim=1] a = triangle[0]
#     cdef cnp.ndarray[DTYPE_t, ndim=1] b = triangle[1]
#     cdef cnp.ndarray[DTYPE_t, ndim=1] c = triangle[2]
#     cdef cnp.ndarray[DTYPE_t, ndim=1] e_1 = b - a
#     cdef cnp.ndarray[DTYPE_t, ndim=1] e_2 = c - a
#     cdef cnp.ndarray[DTYPE_t, ndim=2] d_ = end_points - start
#     # cdef cnp.ndarray[DTYPE_t, ndim=1] n = [e_1[1] * e_2[2] - e_1[2] * e_2[1],
#     #                                        e_1[2] * e_2[0] - e_1[0] * e_2[2],
#     #                                        e_1[0] * e_2[1] - e_1[1] * e_2[0]]
#     cdef cnp.ndarray[DTYPE_t, ndim=1] n_vec
#     # n_vec = np.cross(e_1, e_2)
#     cdef cnp.ndarray[DTYPE_t, ndim=1] r = start - a
#
#     cdef double PARALLEL_THRESHOLD = 1e-6
#     cdef cnp.ndarray[DTYPE_t, ndim=1] condition = np.zeros(_n, dtype=np.bool)
#
#     cdef int i
#     cdef cnp.ndarray[DTYPE_t, ndim=1] d_i
#     cdef double dot_product, t, u, v
#     cdef cnp.ndarray[DTYPE_t, ndim=2] matrix
#     cdef cnp.ndarray[DTYPE_t, ndim=1] result
#
#     for i in range(_n):
#         d_i = d_[i]
#         dot_product = np.dot(n_vec, d_i)
#         if fabs(dot_product) > PARALLEL_THRESHOLD:
#             matrix = np.array([-d_i, e_1, e_2]).T
#             result = np.linalg.solve(matrix, r)
#             t, u, v = result[0], result[1], result[2]
#             if (0 < t <= 1) and (u >= 0) and (v >= 0) and (u + v <= 1):
#                 condition[i] = True
#
#     return condition
