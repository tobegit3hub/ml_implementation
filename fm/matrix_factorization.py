#!/usr/bin/python

import numpy


def matrix_factorization(R, K=10):
  """
  Args:
    R: the objetive [N, M] matrix, P * Q => R
    K: the internal feature number

  Return:
    P: the trained P
    Q: the trained Q
  """

  train_step = 1000
  learning_rate = 0.01
  regular_beta = 0.01

  N = len(R)
  M = len(R[0])
  P = numpy.random.rand(N, K)
  Q = numpy.random.rand(K, M)

  # Train with specified steps
  for step in xrange(train_step):

    # Loop the 2-dim matrix
    for i in xrange(N):
      for j in xrange(M):

        # Ignore the missing data
        if R[i][j] > 0:

          # Compute the (r - r'), r' = Pi * Qj
          eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])

          # Update P and Q with gradient
          for k in xrange(K):

            # Refer to https://mp.weixin.qq.com/s/CD6TrQeKOkGZkbd7Zklaqg , grad = -2 * (r - r') * Q
            P[i][k] = P[i][k] + learning_rate * (
                2 * eij * Q[k][j] - regular_beta * P[i][k])
            Q[k][j] = Q[k][j] + learning_rate * (
                2 * eij * P[i][k] - regular_beta * Q[k][j])

  return P, Q


if __name__ == "__main__":
  R = [
      [5, 3, 0, 1],
      [4, 0, 0, 1],
      [1, 1, 0, 5],
      [1, 0, 0, 4],
      [0, 1, 5, 4],
  ]
  R = numpy.array(R)
  print("Orgin R: {}".format(R))

  generated_P, generated_Q = matrix_factorization(R, 10)
  generated_R = numpy.dot(generated_P, generated_Q)
  print("P: {}\nQ: {}\nR: {}".format(generated_P, generated_Q, generated_R))
