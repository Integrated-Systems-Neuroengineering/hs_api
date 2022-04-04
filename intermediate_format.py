def read_conn_file(conn_file_path, print_to_screen = False):
  '''
  Parse out axon and neuron addresses and synaptic weights from a given connectivity text file
  :param conn_file_path: path to connectivity file to be parsed
  :type conn_file_path: str
  :param print_to_screen: write all parsed axon and neuron weight tuples to screen
  :type print_to_screen: bool, optional
  :return: list of axon weights formatted as 
    [(int) pre-synaptic axon address (external input), (int) post-synaptic neuron address, 
      (float) synaptic weight for this connection] and neuron weights (similar to axons) 
    as a list of [(int) pre-synaptic neuron address, (int) post-synaptic neuron address, 
      (float) synaptic weight for this connection].
  '''
  # Delimiters used in connectivity file
  delimiters = ':', ',', ' ', '[',']','(',')','\n'
  regexPattern = '|'.join(map(re.escape, delimiters)) # generate a regex pattern

  # Containers for collecting axon and neuron addresses/weights
  axons = []
  neurons = []

  # Flow control for parsing
  parse_axons = False
  parse_neurons = False
  with open(conn_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
      #print(line)

      # determine which section is beginning
      if 'Axons' in line: # parse for external inputs to the network
        parse_axons = True
        parse_neurons = False
      elif 'Neurons' in line: # parse for neuron-neuron synapses internal to the network
        parse_axons = False
        parse_neurons = True
      else: # parse out axons and neurons
        # Remove all delimiters
        line = re.split(regexPattern, line)
        #print(line)
        anw = [s for s in line if s!=''] # is it an address?
        #print(anw)
    
        if parse_axons:
          a_i = int(anw[0]) # get the axon (input) ndx
          n_j = [int(s) for s in anw[1::2]]
          w_j = [float(s) for s in anw[2::2]]
          for j in range(len(w_j)):
            axons.append([a_i, n_j[j], w_j[j]])

        elif parse_neurons: # same code, just different param names
          n_i = int(anw[0]) # get the axon (input) ndx
          n_j = [int(s) for s in anw[1::2]]
          w_j = [float(s) for s in anw[2::2]]
          for j in range(len(w_j)):
            neurons.append([n_i, n_j[j], w_j[j]])

  if print_to_screen:
    print('-'*5, 'Found Axons', '-'*5)
    for axon in axons:
      print(axon)

    print('-'*5, 'Found Neurons', '-'*5)
    for neuron in neurons:
      print(neuron)

  return axons, neurons

def read_input_file(input_file_path, B, print_to_screen = False):
  '''
  Read axon inputs from file and return as Mx<num_steps> numpy array
  :param input_file_path: path to spike (Iext) input file to be parsed
  :type input_file_path: str
  :param B: axon weight matrix (to determine number of axons)
  :type B: numpy array
  :param print_to_screen: print parsed spike times to screen
  :type print_to_screen: bool, optional
  :return: array of spike times per axon
  :rtype: numpy array
  '''
  M, N = B.shape # get number of axons

  # Delimiters used in input file (same as connectivity file)
  delimiters = ':', ',', ' ', '[',']','(',')','\n'
  regexPattern = '|'.join(map(re.escape, delimiters)) # generate a regex pattern

  with open(input_file_path, 'r') as f:
    lines = f.readlines()
    num_steps = int(lines[-1].split(':')[0]) +1 # get number of time steps from last row
    print('#axons (M) = {}, #time steps = {}'.format(M, num_steps))

    Iext = np.zeros(shape=(M, num_steps))

    for line in lines:
      #print(line)
      
      # Remove all delimiters
      line = re.split(regexPattern, line)
      #print(line)
      active_axons = [int(s) for s in line if s!=''] # is it an address?
      t = active_axons[0]
      for spiking_axon in active_axons[1:]:
        Iext[spiking_axon][t] = 1
    
  if print_to_screen:
    print(Iext)
  
  return Iext

def conn_to_numpy(axons, neurons):
  '''
  Convert lists outputted by parse_conn_file to weight matrices
  :param axons: output of parse_conn_file as list of 
      [pre-synapic axon, post-synaptic neuron, weight] 
  :type axons: list
  :param neurons: output of parse_conn_file as list of 
      [pre-synapic neuron, post-synaptic neuron, weight] 
  :type neurons: list
  :return: MxN axon weights (B), and NxN weight matrix (W)
  :rtype: tuple of numpy arrays
  '''
  axons = np.array(axons) # this casts all data to float
  neurons = np.array(neurons)

  # Get the number of axons and neurons used
  M = int(max(axons[:, 0])) +1 # also count 0th axon
  N = int(np.amax(neurons[:, :2])) +1 # amax finds the max element in an array
  print('M (#axons) = {}, N (#neurons) = {}'.format(M, N))

  # Get the axon inputs
  B = np.zeros(shape=(M, N))
  for ax in axons:
    ai = int(ax[0])
    nj = int(ax[1])
    B[ai][nj] = ax[2] # synaptic weight
    #print(B)

  # Get the weight matrix
  W = np.zeros(shape=(N, N))
  for n in neurons:
    i = int(n[0])
    j = int(n[1])
    W[i][j] = n[2] # synaptic weight
    #print(W)

  return B, W
 
def write_conn_file(B, W, 
                    opath = './test_connectivity.txt', 
                    print_to_screen = True):
  '''
  Write connectivity text file to disk, and optionally print the same to screen
  :param B: array of axon-neuron synaptic strengths
  :type B: MxN numpy array
  :param W: neuron-neuron synaptic strengths (weight matrix) 
  :type W: NxN numpy array
  :param opath: file path to write connectivity file to
  :type opath: str, optional
  :param print_to_screen: print buffer written to py console (in addition to writing file)
  :type print_to_screen: bool, optional
  :return: None
  '''
  # Determine the number of axons (M) and neurons (N)
  M, N = B.shape
  NW1, NW2 = W.shape

  # Check matrix dimensions
  if NW1 != N:
    print('Error: different numbers of neurons defined in axon [{}x{}] and neuron [{}x{}] weight matrices!'.format(
        M, N, NW1, NW2
    ))
  elif NW1 != NW2:
    print('Error: neuron weight matrix must be square, but its [{}x{}].'.format(
        NW1, NW2
    ))
  else: # no errors
    # Collect axon weights
    buffer = []
    buffer.append('Axons')
    for i in range(M):
      out = '{}: '.format(i)
      edges = [] # container for collecting all (neuron, weight) pairs
      for j in range(N):
        weight = B[i][j]
        if weight != 0: # only list non-zero weights
          edges.append('({}, {})'.format(j, B[i][j]))
      out += '[' + ', '.join(edges) + ']'
      buffer.append(out)

    # Collect neural weights
    buffer.append('Neurons')
    for i in range(N):
      out = '{}: '.format(i)
      edges = [] # container for collecting all (pre-synaptic neuron, post-synaptic neuron, weight) tuples
      for j in range(N):
        weight = W[i][j]
        if weight != 0: # only list non-zero weights
          edges.append('({}, {})'.format(j, weight))
      out += '[' + ', '.join(edges) + ']'
      buffer.append(out)

    # Write connectivity file
    with open(opath, 'w') as f:
      for line in buffer:
        f.write(line+'\n')
    print('Wrote connectivity file to:', opath)

    # Optionally write contents to console
    if print_to_screen:
      print('-'*5, 'Wrote', '-'*5)
      for line in buffer:
        print(line)

def write_input_file(Iext,
                     opath = './test_inputs.txt', 
                     print_to_screen = True):
  '''
  Write connectivity text file to disk, and optionally print the same to screen
  :param Iext: array of axon spike times
  :type Iext: Mx<#time steps> numpy array
  :param opath: file path to write input spike file to
  :type opath: str, optional
  :param print_to_screen: print buffer written to py console (in addition to writing file)
  :type print_to_screen: bool, optional
  :return: None
  '''
  # Determine #axons and #time steps from spike input array
  M, num_steps = Iext.shape

  buffer = []
  for n in range(num_steps):
    out = '{}: '.format(n)
    #print(Iext[:, n])

    spiking_axons = np.where(Iext[:, n] == 1)[0]
    #print(spiking_axons)

    #if len(spiking_axons) > 0:
    spiking_axons = [str(x) for x in spiking_axons]
    out += '[' + ', '.join(spiking_axons) + ']'
    buffer.append(out)

  # Write connectivity file
  with open(opath, 'w') as f:
    for line in buffer:
      f.write(line+'\n')
  print('Wrote input file to:', opath)

  # Optionally write contents to console
  if print_to_screen:
    print('-'*5, 'Wrote', '-'*5)
    for line in buffer:
      print(line)
