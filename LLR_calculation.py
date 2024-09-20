import numpy as np
import matplotlib.pyplot as plt
import math

#Different mapping for the QAMs

def zero_qam64 (value):
    array = []
    if (value == 0):
        for real_part in [3, 1, 5, 7]:
            for imag_part in [5, 7, 3, 1, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 1):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [5, 7, 3, 1]:
                array.append(real_part + 1j * imag_part)
    if (value == 2):
        for real_part in [3, 1, -3, -1]:
            for imag_part in [3, 1, 5, 7, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 3):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [3, 1, -3, -1]:
                array.append(real_part + 1j * imag_part)
    if (value == 4):
        for real_part in [3, 5, -3, -5]:
            for imag_part in [3, 1, 5, 7, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 5):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [3, 5, -3, -5]:
                array.append(real_part + 1j * imag_part)
    return array
                
                
def one_qam64 (value):
    array = []
    if (value == 0):
        for real_part in [-3, -1, -5, -7]:
            for imag_part in [5, 7, 3, 1, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 1):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [-5, -7, -3, -1]:
                array.append(real_part + 1j * imag_part)
    if (value == 2):
        for real_part in [5, 7, -5, -7]:
            for imag_part in [3, 1, 5, 7, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 3):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [5, 7, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 4):
        for real_part in [1, 7, -1, -7]:
            for imag_part in [3, 1, 5, 7, -3, -1, -5, -7]:
                array.append(real_part + 1j * imag_part)
    if (value == 5):
        for real_part in [3, 1, 5, 7, -3, -1, -5, -7]:
            for imag_part in [1, 7, -1, -7]:
                array.append(real_part + 1j * imag_part)
    return array
                
def zero_qam256 (value):
    array = []
    if (value == 0):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 1):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15]:
                array.append(real_part + 1j * imag_part)
    if (value == 2):
        for real_part in [5, 7, 3, 1, -5, -7, -3, -1]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 3):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [5, 7, 3, 1, -5, -7, -3, -1]:
                array.append(real_part + 1j * imag_part)
    if (value == 4):
        for real_part in [5, 7, 11, 9, -5, -7, -11, -9]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 5):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [5, 7, 11, 9, -5, -7, -11, -9]:
                array.append(real_part + 1j * imag_part)
    if (value == 6):
        for real_part in [5, 3, 11, 13, -5, -3, -11, -13]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 7):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [5, 3, 11, 13, -5, -3, -11, -13]:
                array.append(real_part + 1j * imag_part)
    return array
def one_qam256 (value):
    array = []
    if (value == 0):
        for real_part in [-5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 1):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [-5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 2):
        for real_part in [11, 9, 13, 15, -11, -9, -13, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 3):
        for real_part in [11, 9, 13, 15, -11, -9, -13, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 4):
        for real_part in [3, 1, 13, 15, -3, -1, -13, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 5):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [3, 1, 13, 15, -3, -1, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 6):
        for real_part in [7, 1, 9, 15, -7, -1, -9, -15]:
            for imag_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
                array.append(real_part + 1j * imag_part)
    if (value == 7):
        for real_part in [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]:
            for imag_part in [7, 1, 9, 15, -7, -1, -9, -15]:
                array.append(real_part + 1j * imag_part)
    return array

#Hardcoded for QAM16 the values for summation
def zero_value (value):
    array = []
    if (value == 0):
        array.append (1 + 3j)
        array.append (1 + 1j)
        array.append (1 - 3j)
        array.append (1 - 1j)
        array.append (3 + 3j)
        array.append (3 + 1j)
        array.append (3 - 3j)
        array.append (3 - 1j)
    elif (value == 1):
        array.append (-1 + 1j)
        array.append (1 + 1j)
        array.append (3 + 1j)
        array.append (-3 + 1j)
        array.append (-1 + 3j)
        array.append (1 + 3j)
        array.append (3 + 3j)
        array.append (-3 + 3j)
    elif (value == 2):
        array.append (1 + 3j)
        array.append (1 + 1j)
        array.append (1 - 3j)
        array.append (1 - 1j)
        array.append (-1 + 3j)
        array.append (-1 + 1j)
        array.append (-1 - 3j)
        array.append (-1 - 1j)
    elif (value == 3):
        array.append (-1 + 1j)
        array.append (1 + 1j)
        array.append (3 + 1j)
        array.append (-3 + 1j)
        array.append (-1 - 1j)
        array.append (1 - 1j)
        array.append (3 - 1j)
        array.append (-3 - 1j)
        
    return array

def one_value (value):
    array = []
    if (value == 0):
        array.append (-1 + 3j)
        array.append (-1 + 1j)
        array.append (-1 - 3j)
        array.append (-1 - 1j)
        array.append (-3 + 3j)
        array.append (-3 + 1j)
        array.append (-3 - 3j)
        array.append (-3 - 1j)
    elif (value == 1):
        array.append (-1 - 1j)
        array.append (1 - 1j)
        array.append (3 - 1j)
        array.append (-3 - 1j)
        array.append (-1 - 3j)
        array.append (1 - 3j)
        array.append (3 - 3j)
        array.append (-3 - 3j)
    elif (value == 2):
        array.append (3 + 3j)
        array.append (3 + 1j)
        array.append (3 - 3j)
        array.append (3 - 1j)
        array.append (-3 + 3j)
        array.append (-3 + 1j)
        array.append (-3 - 3j)
        array.append (-3 - 1j)
    elif (value == 3):
        array.append (-1 + 3j)
        array.append (1 + 3j)
        array.append (3 + 3j)
        array.append (-3 + 3j)
        array.append (-1 - 3j)
        array.append (1 - 3j)
        array.append (3 - 3j)
        array.append (-3 - 3j)
        
    return array
    
#initialization - no need to change any values while testing
#Automatically changes when modutype and VAR are changed.
number_bits = 0
A = 1
maxX = 1

#only variables that need to be changed in testing to see different outputs
modutype = 'qam16' 
VAR = 0.01

if modutype == 'qam256':
    number_bits = 8
    A = np.sqrt (170)
    maxX = 15
elif modutype == 'qam16':
    number_bits = 4
    A = np.sqrt (10)
    maxX = 4
elif modutype == 'qam64':
    number_bits = 6
    A = np.sqrt (42)
    maxX = 7
    
#making the complex number
x = np.arange(-maxX, maxX + 1) / A
x = x.reshape(-1, 1)
z = x + 1j * x 



#Exponent and then log - Finding the LLR by log map
top = 0 + 0j
bottom = 0 + 0j
errors = []
top_value = 0
bottom_value = 0
for i in range (len (z)):
    for j in range (number_bits):
        top = 0.0
        bottom = 0.0
        returned_array_zero = []
        returned_array_one = []
        if (modutype == 'qam256'):
            returned_array_zero = zero_qam256 (j)
            returned_array_one = one_qam256 (j)
            A = np.sqrt (170)
        elif (modutype == 'qam16'):
            returned_array_zero = zero_value (j)
            returned_array_one = one_value (j)
            A = np.sqrt (10)
        elif (modutype == 'qam64'):
            returned_array_zero = zero_qam64 (j)
            returned_array_one = one_qam64(j)
            A = np.sqrt (42)
            
        for k in range (len (returned_array_zero)):
            real_val = ((returned_array_zero [k].real / A - z[i].real ) ** 2) 
            imaginary = ((returned_array_zero [k].imag / A - z[i].imag) ** 2) 
            top += np.exp (-(real_val[0] + imaginary[0])/VAR )
            real_val2 = ((returned_array_one [k].real / A - z[i].real) ** 2) 
            imaginary2 = ((returned_array_one [k].imag / A - z[i].imag) ** 2) 
            bottom += np.exp(-(real_val2[0] + imaginary2[0])/VAR)
                
        top_real = top
        bottom_real = bottom
        if (top_real != 0 and bottom_real != 0):
            errors.append (np.log (top_real/bottom_real)/100)
        else:
            errors.append (0)
    
    
#plotting the graps    
num_rows = (number_bits + 1) // 2
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(17, 5* num_rows))

for i in range(number_bits):
    row = i // 2
    col = i % 2
    axes[row, col].set_title(f'Complex Plane Plot of L_{i}')
    if i % 2 == 0:
        axes[row, col].set_xlabel('Re(x)')
        axes[row, col].scatter(z.real, errors[i::number_bits], s = 60, c='b', marker='x')
    else:
        axes[row, col].set_xlabel('Im(x)')
        axes[row, col].scatter(z.imag, errors[i::number_bits], s = 60, c='r', marker='x')
    axes[row, col].set_ylabel(f'LLR {i}')
    axes[row, col].grid()

plt.tight_layout()
plt.show()

