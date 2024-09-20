import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras import regularizers
from keras import optimizers
from keras import initializers
'''
GLOBALS: Gray Coded 
'''
QAM_64 = [[4, 12, 28, 20, 52, 60, 44, 36], 
          [5, 13, 29, 21, 53, 61, 45, 37],
          [7, 15, 31, 23, 55, 63, 47, 39],
          [6, 14, 30, 22, 54, 62, 46, 38],
          [2, 10, 26, 18, 50, 58, 42, 34],
          [3, 11, 27, 19, 51, 59, 43, 35],
          [1, 9, 25, 17, 49, 57, 41, 33],
          [0, 8, 24, 16, 48, 56, 40, 32]]


QAM_16 = [[0, 4, 12, 8],
          [1, 5, 13, 9],
          [3, 7, 15, 11],
          [2, 6, 14, 10]]

QAM_4 = [[1, 3],
         [0, 2]]

QAM_64_b = [['000100', '001100', '011100', '010100', '110100', '111100', '101100', '100100'],
            ['000101', '001101', '011101', '010101', '110101', '111101', '101101', '100101'],
            ['000111', '001111', '011111', '010111', '110111', '111111', '101111', '100111'],
            ['000110', '001110', '011110', '010110', '110110', '111110', '101110', '100110'],
            ['000010', '001010', '011010', '010010', '110010', '111010', '101010', '100010'], 
            ['000011', '001011', '011011', '010011', '110011', '111011', '101011', '100011'], 
            ['000001', '001001', '011001', '010001', '110001', '111001', '101001', '100001'], 
            ['000000', '001000', '011000', '010000', '110000', '111000', '101000', '100000']]

QAM_16_b = [['0000', '0100', '1100', '1000'],
           ['0001', '0101', '1101', '1001'],
           ['0011', '0111', '1111', '1011'],
           ['0010', '0110', '1110', '1010']]


QAM_4_b = [['01', '11'],
           ['00', '10']]

QAM_256 = [[ 0,  16,  48,  32, 96, 112,  80,  64, 192, 208, 240, 224, 160, 176, 144, 128],
           [ 1,  17,  49,  33,  97, 113,  81,  65, 193, 209, 241, 225, 161, 177, 145, 129],
           [ 3,  19,  51,  35,  99, 115,  83,  67, 195, 211, 243, 227, 163, 179, 147, 131],
           [ 2,  18,  50,  34,  98, 114,  82,  66, 194, 210, 242, 226, 162, 178, 146, 130],
           [ 6,  22,  54,  38, 102, 118,  86,  70, 198, 214, 246, 230, 166, 182, 150, 134],
           [ 7,  23,  55,  39, 103, 119,  87,  71, 199, 215, 247, 231, 167, 183, 151, 135],
           [ 5,  21,  53,  37, 101, 117,  85,  69, 197, 213, 245, 229, 165, 181, 149, 133],
           [ 4,  20,  52,  36, 100, 116,  84,  68, 196, 212, 244, 228, 164, 180, 148, 132],
           [ 12,  28,  60,  44, 108, 124,  92,  76, 204, 220, 252, 236, 172, 188, 156, 140],
           [ 13,  29,  61,  45, 109, 125,  93,  77, 205, 221, 253, 237, 173, 189, 157, 141],
           [ 15,  31,  63,  47, 111, 127,  95,  79, 207, 223, 255, 239, 175, 191, 159, 143],
           [ 14,  30,  62,  46, 110, 126,  94,  78, 206, 222, 254, 238, 174, 190, 158, 142],
           [ 10,  26,  58,  42, 106, 122,  90,  74, 202, 218, 250, 234, 170, 186, 154, 138],
           [ 11,  27,  59,  43, 107, 123,  91,  75, 203, 219, 251, 235, 171, 187, 155, 139],
           [ 9,  25,  57,  41, 105, 121,  89,  73, 201, 217, 249, 233, 169, 185, 153, 137],
           [ 8,  24,  56,  40, 104, 120,  88,  72, 200, 216, 248, 232, 168, 184, 152, 136]]

QAM_256_2 = [
    [140, 132, 164, 32, 160, 147, 146, 108, 110, 106, 234, 250, 251, 235, 133, 141],
    [12, 4, 36, 0, 128, 151, 150, 109, 238, 42, 170, 186, 187, 171, 163, 143],
    [76, 20, 16, 179, 136, 135, 134, 142, 174, 43, 217, 1, 3, 139, 131, 195],
    [92, 28, 24, 88, 89, 113, 115, 99, 46, 47, 15, 9, 11, 138, 130, 175],
    [126, 60, 124, 120, 121, 241, 243, 227, 14, 30, 31, 8, 10, 152, 208, 144],
    [62, 61, 239, 122, 123, 55, 183, 247, 215, 26, 27, 72, 74, 184, 216, 220],
    [63, 199, 231, 167, 127, 119, 181, 245, 213, 68, 64, 65, 78, 223, 248, 252],
    [59, 51, 35, 166, 125, 117, 53, 244, 212, 84, 80, 81, 94, 95, 232, 236],
    [58, 50, 34, 162, 85, 21, 149, 148, 249, 116, 112, 19, 83, 87, 168, 172],
    [56, 54, 38, 221, 93, 5, 69, 71, 2, 6, 22, 18, 82, 210, 218, 90],
    [57, 185, 189, 253, 29, 13, 77, 67, 66, 70, 86, 118, 114, 211, 219, 91],
    [156, 157, 191, 255, 25, 153, 178, 176, 177, 49, 33, 214, 246, 155, 192, 196],
    [158, 159, 200, 97, 104, 190, 182, 48, 145, 17, 204, 222, 254, 240, 224, 228],
    [154, 233, 201, 73, 105, 188, 180, 52, 209, 197, 205, 237, 173, 242, 226, 230],
    [194, 202, 203, 75, 107, 40, 41, 169, 193, 225, 229, 7, 39, 103, 102, 98],
    [198, 206, 207, 79, 111, 44, 45, 137, 129, 161, 165, 23, 37, 101, 100, 96]
]





QAM_256_b= [
    [bin(value)[2:].zfill(8) for value in row] for row in QAM_256
]

         
'''
Part 1:
Generate Data for 4QAM, 16QAM, 64QAM, BPSK, 8PSK
i. Generate stream of bits c
ii. Divide c into M-sized chunks
iii. Map each M-sized chunk into constellation vector, s (Dim N)
iv. Add AWGN to s
'''

class System():
  def __init__(self, num_bits_send, modulation):
    self.num_bits_send = num_bits_send
    self.type = modulation 
    self.snr = None; self.N_0 = None; self.llr_ = None 

    if modulation == '4QAM':
      self.M = 4; self.k = 2; self.binary_matrix = QAM_4_b; self.A = np.sqrt (2)
    elif modulation == '16QAM':
      self.M = 16; self.k = 4; self.binary_matrix = QAM_16_b; self.A = np.sqrt (10)
    elif modulation == '64QAM' :
      self.M = 64; self.k = 6; self.binary_matrix = QAM_64_b; self.A = np.sqrt(42)
    elif modulation == '256QAM': 
      self.M = 256; self.k = 8; self.binary_matrix = QAM_256_b; self.A = np.sqrt(170)
  
  def send_n_receive(self, snr):
    self.snr = snr
    print('Sending %d bits with snr = %fdB' %(self.num_bits_send, snr))
    bit_stream = self.binary_data(self.num_bits_send, self.M)
    self.bit_stream = bit_stream
    y = self.bit_stream_to_grid(bit_stream, type = self.type)
    self.N_0 = self.snr_to_N0(snr, type = self.type)
    r = self.add_noise(y, self.N_0/2)
    self.r = r
    d = self.decoder(r, self.k, self.binary_matrix, self.N_0)
    self.num_error = np.sum(np.abs(bit_stream-d))
    self.b_error = self.num_error/self.num_bits_send
    print('Number of Bit Errors %f \nBit Error Rate: %f' %(self.num_error, self.num_error/self.num_bits_send))

  def generate_bits(self, n):
    '''
    n: number of bits to randomly generate from uniform distribution
    ret: n,1 array of bits
    '''
    return np.around(np.random.rand(n,1))

  def divide_to_k(self, c, k):
    '''
    c: n,1 array of bits
    k: size of each chunk (i.e. M = 2**k)
    ret: k, n/k array of bits
    '''
    n, d = np.shape(c)
    if n%k == 0:
      copy = np.transpose(c.reshape((int(n/k), k)))
      return copy
    else:
      copy = np.append(c, np.zeros((k-n%k,1)))
      n_, = np.shape(copy)
      copy = np.transpose(copy.reshape((int(n_/k), k)))  
    return copy

  def binary_data(self, n, M):
    '''
    n: number of bits to randomly generate
    M: (int) type of modulation
    ret: k, n/k array of bits
    '''
    return self.divide_to_k(self.generate_bits(n), int(np.log2(M)))

  def bits_to_base10(self, c):
    '''
    c: n,1 array of bits. eg. [0; 1; 0; 1] = 0101 = 5
    ret: float, decimal conversion of c
    '''
    n, = np.shape(c)
    temp = 2**np.arange(n-1, -1, -1)
    return np.dot(c, temp)

  def serial_parallel_converter(self, c, type = '4QAM'):
    '''
    c: n,1 array of bits
    ret: tuple, constellation coordinates -> CHANGE TO GRAY CODE
    '''
    
    n, = np.shape(c)
    num = int(self.bits_to_base10(c))
    
    x1 = 0; y1 = 0
    if type == '4QAM':
        for i in range(2):
            for j in range(2):
                 if QAM_4[i][j] == num:
                    y1 = 1 - 2*i; x1 = -1 + 2*j
    elif type == '16QAM':
        for i in range(4):
            for j in range(4):
                if QAM_16[i][j] == num:
                    y1 = 3 - 2*i; x1 = -3 + 2*j
    elif type == '64QAM':
        for i in range(8):
            for j in range(8):
                if QAM_64[i][j] == num:
                  y1 = 7 - 2*i; x1 = -7 + 2*j
    elif type == '256QAM':
        for i in range(16):
            for j in range(16):
                if QAM_256[i][j] == num:
                    y1 = 15 - 2*i; x1 = -15 + 2*j
    return x1, y1

  def bit_stream_to_grid(self, bit_stream, type = '256QAM'):
    '''
    bit_stream: k, n/k array of bits (divided into chunks)
    ret: n/k, 2 array of constellation coordinates
    '''
    n, d = np.shape(bit_stream)
    for i in range(d):
      x1, x2 = self.serial_parallel_converter(bit_stream[:, i], type)
      if i == 0:
        y = np.array([[x1, x2]])
      else:
        y = np.append(y, np.array([[x1, x2]]), axis = 0)
    return y

  def add_noise(self, y, var):
    '''
    y: n,d array
    var: float, variance
    ret: n,d array with variance var AWGN
    '''
    n, d = np.shape(y)
    return y + math.sqrt(var)*np.random.randn(n, d)

  def snr_to_N0(self, snr_db, type = '4QAM'):
    snr = 10**(snr_db/10)
    if type == '4QAM':
        bit_num = 2
    elif type == '16QAM':
        bit_num = 4
    elif type == '64QAM':
        bit_num = 6
    elif type == '256QAM':
        bit_num = 8
    else:
        bit_num = 0
    print(type)
      
    temp = np.arange(0, 2**(bit_num/2))
    amp_list = 2*(temp - np.average(temp))
    n, = np.shape(temp)
    sum = 0 
    for i in range(n):
        for j in range(n):
            sum += amp_list[i]**2 + amp_list[j]**2
    e_avg = sum/2**bit_num
    return e_avg/(bit_num*snr)

  def dec_to_bin(self, x, n):
      s = bin(x)[2:]; temp = ''
      if len(s) < n:
        for i in range(int(n/2) - len(s)):
          temp += '0'
        s = temp + s
      return s

  def find_coordinates(self, target_index, binary_matrix):
    order = len(binary_matrix) 
    mat_index = len(binary_matrix[0][0]) - target_index - 1
    zero_mat = np.zeros((int((order**2)/2), 2)); zero_cnt = 0
    one_mat = np.zeros((int((order**2)/2), 2)); one_cnt = 0
    for i in range(order):
      for j in range (len(binary_matrix[0])):
          if binary_matrix[i][j][mat_index] == '0':
            zero_mat[zero_cnt, 1] = order-1 - 2*i
            zero_mat[zero_cnt, 0] = -1*order+1 + 2*j
            zero_cnt += 1
          else:
            one_mat[one_cnt, 1] = order-1 - 2*i
            one_mat[one_cnt, 0] = -1*order+1 + 2*j
            one_cnt += 1
    return zero_mat, one_mat

  def r_to_llr(self, r, bit_num, binary_matrix, N_0):
    zero_sum = 0; one_sum = 0
    A = self.A
    for i in range(int((len(binary_matrix)**2)/2)):
      if i == 0:
        r_ = np.array([np.copy(r)])
      else:
        r_ = np.append(r_, np.array([r]), axis = 0)
    li = []
    for i in range(bit_num):
      zero_mat, one_mat = self.find_coordinates(i, binary_matrix)
      z_norm = (((r_ - zero_mat))[:, 0])**2 / A + (((r_ - zero_mat))[:, 1])**2 / A
      o_norm = (((r_ - one_mat))[:, 0])**2 / A + (((r_ - one_mat))[:, 1])**2 / A
      top = np.sum(np.exp((-2/(N_0))*z_norm))
      bottom = np.sum(np.exp((-2/(N_0))*o_norm))
      if (top!= 0 and bottom!= 0):
         li.append(np.log(top/(bottom)))
      else:  
         li.append(2/(N_0)*(np.min(o_norm)-np.min(z_norm)))
    return li

  def r_to_llr_approx(self, r, bit_num, binary_matrix, N_0):
    zero_sum = 0; one_sum = 0
    for i in range(int((len(binary_matrix)**2)/2)):
      if i == 0:
        r_ = np.array([np.copy(r)])
      else:
        r_ = np.append(r_, np.array([r]), axis = 0)
    li = []
    for i in range(bit_num):
      zero_mat, one_mat = self.find_coordinates(i, binary_matrix)
      z_norm = ((r_ - zero_mat)*(r_ - zero_mat))[:, 0] + ((r_ - zero_mat)*(r_ - zero_mat))[:, 1]
      o_norm = ((r_ - one_mat)*(r_ - one_mat))[:, 0] + ((r_ - one_mat)*(r_ - one_mat))[:, 1]
      li.append(2/(N_0)*(np.min(o_norm)-np.min(z_norm)))
    return li

  def decoder(self, r, bit_num, binary_matrix, N_0):
    n, d = np.shape(r)
    for i in range(n):
      llr = self.r_to_llr(r[i, :], bit_num, binary_matrix, N_0)
      llr2 = self.r_to_llr_approx (r[i, :], bit_num, binary_matrix, N_0)
      llr2.reverse()
      llr.reverse()           
      if self.llr_ == None:
        self.llr = np.transpose(np.array([np.asarray(llr)])); self.llr_ = 1
      else:
        self.llr = np.append(self.llr, np.transpose(np.array([np.asarray(llr)])), axis = 1)
        
      llr = np.asarray(llr)
      zero_indices = np.where(llr == 0)[0]
      llr_normalized = 0.5 * (-1 * llr / np.abs(llr) + 1)
      llr_normalized[zero_indices] = 0
      llr = llr_normalized
    
      if i == 0:
        ans = np.transpose(np.array([llr]))
      else:
        ans = np.append(ans, np.transpose(np.array([llr])), axis = 1)
    return ans


    
    
class LLR_net():
  def __init__(self, modulation, training_size, test_size, train_snr=10, test_snr=10, epoch_size=400):
    self.training_system = System(training_size, modulation)
    self.test_system = System(test_size, modulation)
    self.training_system.send_n_receive(train_snr)
    self.test_system.send_n_receive(test_snr)
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(self.test_system.k, activation='linear'))
    self.model = model
    self.epoch_size = int(epoch_size)
  
  def train(self):
    X = self.training_system.r
    y = np.transpose(self.training_system.llr)
    self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    self.model.fit(X, y, epochs=self.epoch_size, batch_size=25, verbose =0 ) 
    
  
  def test(self):
    X = self.test_system.r
    y = np.transpose(self.test_system.llr)
    self.predictions = self.model.predict(X)
    self.decode = np.transpose(0.5*(-1*self.predictions/np.abs(self.predictions)+1))
    self.num_error = np.sum(np.abs(self.test_system.bit_stream-self.decode))
    self.b_error = self.num_error/self.test_system.num_bits_send
    self.conventional_error = self.test_system.b_error
    self.llr_per_bit = self.test_system.llr
    self.real_part = self.test_system.r.real
    print('LLR Net bit error rate is %f' %(self.b_error))
   
  def plot_llr_per_bit(self):
    # Assuming you have already run the test method and stored the LLR values in self.llr_per_bit
    # Assuming you have already run the test method and stored the necessary values
    llr_values = self.llr_per_bit
    real_part = self.test_system.r.real

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    print (len(real_part))
    print (len (llr_values))

    # Plot the scatter plot
    ax.scatter(real_part, llr_values)

    # Set the title and axis labels
    ax.set_title("LLR Values vs Real Part of Modulation")
    ax.set_xlabel("Real Part of Modulation")
    ax.set_ylabel("LLR Value")

    # Show the plot
    plt.show()
    
llr_total = []
llr_total2 = []
llr_total3 = []
x  = []
snr_list =[-5, 0, 5, 10, 20]
llr_conventional = []; conventional_error = [];
for snr in snr_list:
  a = LLR_net('256QAM', 20000, 1000, train_snr = 5, test_snr = snr)
  a.train()
  a.test()
  llr_conventional.append(a.b_error)
  conventional_error.append(a.conventional_error)
  llr_total.append (a.llr_per_bit[0])
  llr_total2.append (a.llr_per_bit[1])
  llr_total3.append (a.llr_per_bit[2])
  print('\n')
  real_part = a.real_part
    
    

  x = llr_total[0]
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.scatter(real_part [:,0], -x)

    # Set the title and axis labels
  ax.set_title("LLR Values vs Real Part of Modulation")
  ax.set_xlabel("Real Part of Modulation")
  ax.set_ylabel("LLR Value")

    # Show the plot
  plt.show()

  y = llr_total2[0]
  fig2, ax2 = plt.subplots(figsize=(12, 6))
  ax2.scatter(real_part [:,0], -y)

    # Set the title and axis labels
  ax2.set_title("LLR Values vs Real Part of Modulation")
  ax2.set_xlabel("Real Part of Modulation")
  ax2.set_ylabel("LLR Value")

    # Show the plot
  plt.show()
  z = llr_total3[0]
  fig3, ax3 = plt.subplots(figsize=(12, 6))
  ax3.scatter(real_part [:,0], -z)

    # Set the title and axis labels
  ax3.set_title("LLR Values vs Real Part of Modulation")
  ax3.set_xlabel("Real Part of Modulation")
  ax3.set_ylabel("LLR Value")

    # Show the plot
  plt.show()

  llr_total= []
  llr_total2 = []
  llr_total3 = []
    
plt.plot(snr_list, llr_conventional)
plt.plot(snr_list, conventional_error)