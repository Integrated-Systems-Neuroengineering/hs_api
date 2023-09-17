"""Cubes

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Fj5HQJG0frh5canLJm38BJtm-b2GJrL

# Importing a library that is not in Colaboratory

To import a library that's not in Colaboratory by default, you can use `!pip install` or `!apt-get install`.
"""
from hs_api.api import CRI_network
import sys
import subprocess
import time
import numpy as np

def computeRes(connections,swResult):
  sum = 0
  for i in connections.keys():
    if i in swResult: #if i spiked
      for entry in connections[i]: #scan through post-synaptic neurons
        j = entry[0]
        w = entry[1]
        if j in swResult: #if post synaptic neuron spiked
          sum = sum + w #add synapse weight to sum
  return sum


def main():
  #parse text file
  import re
  with open('G15') as f: lines = [line.rstrip() for line in f]

  dat = [[int(e) for e in re.findall('\d+', str)] for str in lines]
  info = dat[0]
  dat = dat[1:]



  connections = {} #declare dict to hold neuorns



  for x in dat:
#      breakpoint()
      i = x[0] #presynaptic neuron
      j = x[1] #postsynaptic neuron
      w = x[2] #synapse weight
      if i in connections: #if presynaptic neuron already in dict
          connections[i].append((j,-w)) #add synapse to post synaptic neuron with negative weight
      else: #if sypnapes isn't in dictionary
          connections[i] = [(j,-1*w)] #add synapse to dictionary and add neuron
      #add symetric synapse
      if j in connections:
          connections[j].append((i,-w))
      else:
          connections[j] = [(i,-w)]

  #add recurent synapse to each neuron
  for i in connections.keys(): #for each neuron
      connect = connections[i] #get synapses
      reccurentSum = sum( [-1*entry[1] for entry  in connect]) #take the sum of the negative of every synapse weight for neuron
      connections[i].append((i,reccurentSum)) #add recurrent connection with sum as weight

  #a single axon is added but never used
  neurons = list(connections.keys())
  axons = {'a' : [(neurons[1],2)]} #add an unused dummy synapse

  #Define a configuration dictionary
  config = {}
  config['neuron_type'] = "I&F" #redundent
  config['global_neuron_params'] = {}
  config['global_neuron_params']['v_thr'] = 0 #voltage threshold is zero

  synsum = 0
  for key in connections.keys():
    synsum += len(connections[key])
  print(synsum)
  breakpoint()

  leak,perturbMag,annealSteps,stepSize = (0, 20, 50, 1)

  dictKeys = connections.keys()


  breakpoint()

  softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, outputs = list(connections.keys()), target='CRI', perturbMag=20, leak=0)
  perturbMag=20
  perturbAll = []
  timeAll = []
  resAll = [] #compute result at each step
  perturbAll.append(perturbMag)
  t0 = time.time()
  for i in range(200000000):
      #breakpoint()
      if i !=0 and i%50==0: #every 50 steps reduce noise magnitude by one bit
        if perturbMag > 0: # stop once noise magnitude is zero
          perturbMag = perturbMag - stepSize #reduce by one bit
          softwareNetwork.set_perturbMag(perturbMag) #set perturbation magnitude
          perturbAll.append(perturbMag)
          print('PerturbMag: '+str(perturbMag))
          timeAll.append(time.time()-t0)
          res = computeRes(connections,swResult[0]) # compute result
          print('Result: '+str(res))
          resAll.append(res)
        else:
          timeAll.append(time.time()-t0)
          res = computeRes(connections,swResult[0]) # compute result
          print('Result: '+str(res))
          resAll.append(res)
          break

      swResult = softwareNetwork.step([]) #run step with no external inputs

      #print("software result: ")
      #swResult.sort() #sort and print result
      #print(swResult)
      #print(len(swResult[0]))
      #print(perturbMag)

      #convert result to onehot encoding
      #print( [1 if q in swResult.sort() else 0 for q in list(connections.keys())])

  print(resAll)
  print(perturbAll)
  print(timeAll)

if __name__ == '__main__':
    main()

