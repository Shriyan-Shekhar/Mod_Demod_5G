import numpy as np
import matplotlib.pyplot as plt

def max_log_map_qam16_soft_demodulation(real_symb_in, N0):
    A = 1 / np.sqrt(10)
    x = real_symb_in
    y = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < -2 * A:
            y[i] = 8 * A / N0 * (x[i] + A)
        elif x[i] < 2 * A:
            y[i] = 4 * A / N0 * x[i]
        else:
            y[i] = 8 * A / N0 * (x[i] - A)

    return y

def max_log_map_qam64_soft_demodulation(real_symb_in, N0):
    A = 1 / np.sqrt (42)
    x = real_symb_in
    y = np.zeros_like(x) 

    for i in range(len(x)):
        if x[i] < -6 * A:
            y[i] = 16 * A / N0 * (x[i] + 3 * A)
        elif x[i] < -4 * A:
            y[i] = 12 * A / N0 * (x[i] + 2 * A)
        elif x[i] < -2 * A:
            y[i] = 8 * A / N0 * (x[i] + A)
        elif x[i] < 2 * A:
            y[i] = 4 * A / N0 * x[i]
        elif x[i] < 4 * A:
            y[i] = 8 * A / N0 * x[i]
        elif x[i] < 6 * A:
            y[i] = 12 *  A/ N0 * (x[i] - 2 * A)
        else:
            y[i] = 16 * A / N0 * (x[i] - 3 * A)

    return y

def max_log_map_qam64_soft_demodulation_2(real_symb_in, N0):
    A = 1 / np.sqrt (42)
    x = real_symb_in
    y = np.zeros_like(x) 

    for i in range(len(x)):
        if x[i] < -6 * A:
            y[i] = 8 * A / N0 * (x[i] + 5 * A)
        elif x[i] < -2 * A:
            y[i] = 4 * A / N0 * (x[i] + 4 * A)
        elif x[i] < 0:
            y[i] = 8 * A / N0 * (x[i] + 3 * A)
        elif x[i] < 2 * A:
            y[i] = 8 * A / N0 * (-x[i] + 3 * A)
        elif x[i] < 6 * A:
            y[i] = 4 * A / N0 * (-x[i] + 4 * A)
        else:
            y[i] = 8 * A / N0 * (- x[i] + 5 * A)

    return y

def max_log_map_qam256_soft_demodulation(realSymbIn, N0):
    A = 1/np.sqrt(170)
    X = realSymbIn
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] < -14*A:
            Y[i] = 32*A/N0 * (X[i] + 7*A)
        elif X[i] < -12*A:
            Y[i] = 28*A/N0 * (X[i] + 6*A)
        elif X[i] < -10*A:
            Y[i] = 24*A/N0 * (X[i] + 5*A)
        elif X[i] < -8*A:
            Y[i] = 20*A/N0 * (X[i] + 4*A)
        elif X[i] < -6*A:
            Y[i] = 16*A/N0 * (X[i] + 3*A)
        elif X[i] < -4*A:
            Y[i] = 12*A/N0 * (X[i] + 2*A)
        elif X[i] < -2*A:
            Y[i] = 8*A/N0 * (X[i] + A)
        elif X[i] < 2*A:
            Y[i] = 4*A/N0 * X[i]
        elif X[i] < 4*A:
            Y[i] = 8*A/N0 * (X[i] - A)
        elif X[i] < 6*A:
            Y[i] = 12*A/N0 * (X[i] - 2*A)
        elif X[i] < 8*A:
            Y[i] = 16*A/N0 * (X[i] - 3*A)
        elif X[i] < 10*A:
            Y[i] = 20*A/N0 * (X[i] - 4*A)
        elif X[i] < 12*A:
            Y[i] = 24*A/N0 * (X[i] - 5*A)
        elif X[i] < 14*A:
            Y[i] = 28*A/N0 * (X[i] - 6*A)
        else:
            Y[i] = 32*A/N0 * (X[i] - 7*A)
    return Y

def max_log_map_qam256_soft_demodulation_1(realSymbIn, N0):
    A = 1/np.sqrt(170)
    X = realSymbIn
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] < -14*A:
            Y[i] = 16*A/N0 * (X[i] + 11*A)
        elif X[i] < -12*A:
            Y[i] = 12*A/N0 * (X[i] + 10*A)
        elif X[i] < -10*A:
            Y[i] = 8*A/N0 * (X[i] + 9*A)
        elif X[i] < -6*A:
            Y[i] = 4*A/N0 * (X[i] + 8*A)
        elif X[i] < -4*A:
            Y[i] = 8*A/N0 * (X[i] + 7*A)
        elif X[i] < -2*A:
            Y[i] = 12*A/N0 * (X[i] + 6*A)
        elif X[i] < 0:
            Y[i] = 16*A/N0 * (X[i] + 5*A)
        elif X[i] < 2*A:
            Y[i] = 16*A/N0 * (-X[i] + 5*A)
        elif X[i] < 4*A:
            Y[i] = 12*A/N0 * (-X[i] + 6*A)
        elif X[i] < 6*A:
            Y[i] = 8*A/N0 * (-X[i] + 7*A)
        elif X[i] < 10*A:
            Y[i] = 4*A/N0 * (-X[i] + 8*A)
        elif X[i] < 12*A:
            Y[i] = 8*A/N0 * (-X[i] + 9*A)
        elif X[i] < 14*A:
            Y[i] = 12*A/N0 * (-X[i] + 10*A)
        else:
            Y[i] = 16*A/N0 * (-X[i] + 11*A)
    return Y

def max_log_map_qam256_soft_demodulation_2(realSymbIn, N0):
    A = 1/np.sqrt(170)
    X = realSymbIn
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] < -14*A:
            Y[i] = 8*A/N0 * (X[i] + 13*A)
        elif X[i] < -10*A:
            Y[i] = 4*A/N0 * (X[i] + 12*A)
        elif X[i] < -8*A:
            Y[i] = 8*A/N0 * (X[i] + 11*A)
        elif X[i] < -6*A:
            Y[i] = 8*A/N0 * (-X[i] - 5*A)
        elif X[i] < -2*A:
            Y[i] = 4*A/N0 * (-X[i] - 4*A)
        elif X[i] < 0:
            Y[i] = 8*A/N0 * (-X[i] - 3*A)
        elif X[i] < 2*A:
            Y[i] = 8*A/N0 * (X[i] - 3*A)
        elif X[i] < 6*A:
            Y[i] = 4*A/N0 * (X[i] - 4*A)
        elif X[i] < 8*A:
            Y[i] = 8*A/N0 * (X[i] - 5*A)
        elif X[i] < 10*A:
            Y[i] = 8*A/N0 * (-X[i] + 11*A)
        elif X[i] < 14*A:
            Y[i] = 4*A/N0 * (-X[i] + 12*A)
        else:
            Y[i] = 8*A/N0 * (-X[i] + 13*A)
    return Y


def qam16_soft_demodulation(symbs_in, N0, method):
    soft_bits = np.zeros((4, len(symbs_in)))

    A = 1 / np.sqrt(10)
    re_x = np.real(symbs_in).ravel()
    im_x = np.imag(symbs_in).ravel()

   
    soft_bits[0, :] = max_log_map_qam16_soft_demodulation(re_x, N0)
    soft_bits[1, :] = max_log_map_qam16_soft_demodulation(im_x, N0)
    soft_bits[2, :] = 4 * A / N0 * (-np.abs(re_x) + 2 * A)
    soft_bits[3, :] = 4 * A / N0 * (-np.abs(im_x) + 2 * A)

    return soft_bits.flatten()

def qam64_soft_demodulation(symbs_in, N0, method):
    soft_bits = np.zeros((4, len(symbs_in)))

    A = 1 / np.sqrt(42)
    re_x = np.real(symbs_in).ravel()
    im_x = np.imag(symbs_in).ravel()

   
    soft_bits[0, :] = max_log_map_qam64_soft_demodulation(re_x, N0)
    soft_bits[1, :] = max_log_map_qam64_soft_demodulation(im_x, N0)
    soft_bits[2, :] = max_log_map_qam64_soft_demodulation_2(re_x, N0)
    soft_bits[3, :] = max_log_map_qam64_soft_demodulation_2(im_x, N0)
    soft_bits[4, :] = 4 * A/N0 * (-abs (-abs(re_x) + 4 * A) + 2 * A)
    soft_bits[5, :] = 4 * A/N0 * (-abs (-abs(im_x) + 4 * A) + 2 * A)

    return soft_bits.flatten()


def qam256_soft_demodulation(symbs_in, N0, method):
    soft_bits = np.zeros((8, len(symbs_in)))

    A = 1 / np.sqrt(170)
    re_x = np.real(symbs_in).ravel()
    im_x = np.imag(symbs_in).ravel()

   
    soft_bits[0, :] = max_log_map_qam256_soft_demodulation(re_x, N0)
    soft_bits[1, :] = max_log_map_qam256_soft_demodulation(im_x, N0)
    soft_bits[2, :] = max_log_map_qam256_soft_demodulation_1(re_x, N0)
    soft_bits[3, :] = max_log_map_qam256_soft_demodulation_1(im_x, N0)
    soft_bits[4, :] = max_log_map_qam256_soft_demodulation_2(re_x, N0)
    soft_bits[5, :] = max_log_map_qam256_soft_demodulation_2(im_x, N0)
    soft_bits[6, :] = 4 * A/N0 * (-abs (-abs(-abs(re_x) + 8 * A) + 4 * A)+ 2 * A)
    soft_bits[7, :] =  4 * A/N0 * (-abs (-abs(-abs(im_x) + 8 * A) + 4 * A)+ 2 * A)

    return soft_bits.flatten()


def nr_soft_modu_demapper(symbs_in, modu_type, N0, method):
    N0 = max(N0, 0.001)

    modu_type = modu_type.lower()
    if modu_type == 'qpsk':
        soft_bits = np.zeros((2, len(symbs_in)))
        A = 1 / np.sqrt(2)
        soft_bits[0, :] = 4 * A / N0 * np.real(symbs_in)
        soft_bits[1, :] = 4 * A / N0 * np.imag(symbs_in)
        soft_bits = soft_bits.flatten()
    elif modu_type == '16qam':
        soft_bits = qam16_soft_demodulation(symbs_in, N0, method)
    elif modu_type == '64qam':
        soft_bits = qam64_soft_demodulation(symbs_in, N0, method)
    elif modu_type == '256qam':
        soft_bits = qam256_soft_demodulation(symbs_in, N0, method)
    else:
        raise ValueError("Invalid modulation type: {}".format(modu_type))

    return soft_bits






maxX = 15  # Adjust this value as needed for specific QAM
A = 1 / np.sqrt (170)    # Adjust this value as needed for QAM
K = 8 #no. of bits


x = np.arange(-maxX, maxX + 1) * A
x = x.reshape(-1, 1)
x_complex = x + 1j * x

moduType = '256qam'  # Adjust this value as needed for QAM type

y1 = nr_soft_modu_demapper(x_complex, moduType, 1, 'max-log-map') #put a method parameter for additional needs - useless for now
y1 = y1.T.reshape(-1, 2 * maxX + 1)


# Create the figure and subplot
fig, ax = plt.subplots(figsize=(12, 8))

#Change the bit number here => for bit 0 make plotting y1[0::] and go up till no. of bits
plotting = []
plotting = y1 [4, :]
# Plot the data
ax.scatter(x , plotting, c = 'b', marker = 'x')
ax.grid(True)
ax.set_xlabel('Re(x)/Im(x) (A)')
ax.set_ylabel(f'LLR (1/N0)')
ax.set_xticks(np.arange(-maxX, maxX + 1, 2))
ax.set_xlim([-2, 2])
ax.legend(['max-log-map'], loc='upper left')

#to plot all, just implement a for loop from 0 to K - 1, so there are k iterations and replace the value in plotting to the variable

# Adjust the layout
plt.tight_layout()
plt.show()