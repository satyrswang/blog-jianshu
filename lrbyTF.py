#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:41:03 2017

@author: wyq
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# linear regression
def main(args):
  # We init as h=x
  W = tf.Variable([1], dtype=tf.float32)
  b = tf.Variable([0], dtype=tf.float32)
  x = tf.placeholder(tf.float32)
  h = W * x + b
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  #sess.run(init)
  #print("hyposis init:", sess.run(h, {x:[1,2,3,4]}))

  y = tf.placeholder(tf.float32)
  squared_deltas = tf.square(h - y)
  cost = 0.5 * tf.reduce_mean(squared_deltas)
  #print("cost init:", sess.run(cost, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

  fixW = tf.assign(W, [-1.])
  fixb = tf.assign(b, [1.])
  sess.run([fixW, fixb])
  #print("W, b, cost expected:", sess.run([fixW, fixb, cost], {x:[1,2,3,4], y:[0,-1,-2,-3]}))

  # linear regression 
  sess.run(init)#assign
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(cost)
  
  for i in range(10000):
    sess.run(train, {x:[1,2,3,4,-3,35], y:[0,-1,-2,-3,4,-34]})
  
  curr_W, curr_b, curr_loss = sess.run([W, b, cost], {x:[1,2,3,4,-3,35], y:[0,-1,-2,-3,4,-34]})
  
  print("W, b, cost learned: ", curr_W, curr_b, curr_loss)


if __name__ == "__main__":
  tf.app.run()
  
