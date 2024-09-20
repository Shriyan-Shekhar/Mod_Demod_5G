import numpy as np
import matplotlib.pyplot as plt
import math

mu = 1
scs = 15e3 * 2**mu
BW = 5e6
num_rb = 11
N = 256
F = scs * N
Ts = 1 / (15e3 * 2048)
CP0 = (144 * 2**-mu + 16) * Ts * F
CP1 = (144 * 2**-mu) * Ts * F
Nsym = 14
A = 5376  # payload length
RB_allocated = 8
Nrb_start = 0
Nsym_allocated = 14
num_ofdm_symbol = 14
Nsym_start = 0
SignalNoiseRatio = 10

snrList = np.arange (1, 21)

#add noise for AWGN
def add_noise(TxSignal, snr):
    data = np.mean(abs(TxSignal**2))
    noise = data/(10**(snr/10))
    overall_noise = 1/np.sqrt(2) * np.sqrt(noise)* (np.random.standard_normal(len(TxSignal)))
    return TxSignal + overall_noise

#calculate and return sd of the noise
def calculate_sd (TxSignal, snr):
    data = np.mean (abs(TxSignal ** 2))
    noise = data/(10**(snr/10))* 10000
    sd = np.sqrt (noise)
    return noise 


#channel for noise - if needed I can add other channel types
def channel(signal, snr, channelType="awgn"):
    if channelType == "awgn":
        outSignal = add_noise(signal, snr)
    return outSignal


# modulating
def modulate (bits, name):
	if name == "QPSK":
		modulation = ((1- 2* bits [ : :2]) + 1j * (1 - 2* bits [1: :2])) / np.sqrt(2)
		return modulation
	elif name == "QAM16":
		modulation = ((1-2*bits[::4])*(2 - (1 - 2 * bits [2::4])) + 1j * (1 - 2 * bits[1::4])*(2 - (1 - 2 * bits [3::4])))/ np.sqrt (10)
		return modulation
	elif name == "QAM64":
		modulation = ((1-2*bits[::6])*(4 - (1 - 2 * bits [2::6]) * (2 - (1 - 2 * bits [4::6]))) + 1j * (1 - 2 * bits[1::6])*(4 - (1 - 2 * bits [3::6])* (2 - (1 - 2 * bits [5::6]))))/ np.sqrt (42)
		return modulation
	elif name == "QAM256":
		modulation =  ((1-2*bits[::8])*(8 - (1 - 2* bits [2::8])*(4 - (1 - 2 * bits [4::8]) * (2 - (1 - 2 * bits [6::8])))) + 1j * (1 - 2 * bits[1::8])*(8 - (1 - 2 * bits [3::8])* (4 - (1 - 2 * bits [5::8])* (2 - (1- 2 * bits [7::8])))))/ np.sqrt (170)
		return modulation

#demondulating the bits and accounting for a small error
def demodulate(signal, name, error):
    if name == "QPSK":
        demod = np.arctan2(np.imag(signal), np.real(signal)) / (np.pi/4)
        demod = (demod > 0).astype(int)
    elif name == "QAM16":
        demod = np.zeros(len(signal) * 4, dtype=int)
        for i in range(len(signal)):
            I = np.real(signal[i] * np.sqrt (10))
            Q = np.imag(signal[i] * np.sqrt (10))
            demod[i*4] = (I <= 0)
            demod[i*4 + 1] = (Q <= 0)
            demod[i*4 + 2] = (abs(I) >= 3 - error)
            demod[i*4 + 3] = (abs(Q) >= 3 - error)
    elif name == "QAM64":
        demod = np.zeros(len(signal) * 6, dtype=int)
        for i in range(len(signal)):
            I = np.real(signal[i] * np.sqrt (42))
            Q = np.imag(signal[i] * np.sqrt (42))
            demod[i*6] = (I <= 0)
            demod[i*6 + 1] = (Q <= 0)
            demod [i*6 + 2] = (abs(I) >= 5 - error)
            demod [i*6 + 3] = (abs(Q) >= 5 - error)
            demod[i*6 + 4] = ((abs(I) >= 1 - error) and (abs (I) <= 1 + error)) or ((abs(I) >= 7 - error) and (abs (I) <= 7 + error))
            demod[i*6 + 5] = ((abs(Q) >= 1 - error) and (abs (Q) <= 1 + error)) or ((abs(Q) >= 7 - error) and (abs (Q) <= 7 + error))
    elif name == "QAM256":
        demod = np.zeros(len(signal) * 8, dtype=int)
        for i in range(len(signal)):
            I = np.real(signal[i] * np.sqrt (170))
            Q = np.imag(signal[i] * np.sqrt (170))
            demod[i*8] = (I <= 0)
            demod[i*8 + 1] = (Q <= 0)
            demod [i*8 + 2] = (abs(I) >= 9 - error)
            demod [i*8 + 3] = (abs(Q) >= 9 - error)
            demod[i*8 + 4] = ((abs(I) >= 3 - error) and (abs (I) <= 3 + error)) or ((abs(I) >= 15 - error) and (abs (I) <= 15 + error))  or ((abs(I) >= 1 - error) and (abs (I) <= 1 + error)) or ((abs(I) >= 13 - error) and (abs (I) <= 13 + error))
            demod[i*8 + 5] = ((abs(Q) >= 3 - error) and (abs (Q) <= 3 + error)) or ((abs(Q) >= 15 - error) and (abs (Q) <= 15 + error))  or ((abs(Q) >= 1 - error) and (abs (Q) <= 1 + error)) or ((abs(Q) >= 13 - error) and (abs (Q) <= 13 + error))
            demod [i*8 + 6] =  ((abs(I) >= 7 - error) and (abs (I) <= 7 + error)) or ((abs(I) >= 1 - error) and (abs (I) <= 1 + error))  or ((abs(I) >= 9 - error) and (abs (I) <= 9 + error)) or ((abs(I) >= 15 - error) and (abs (I) <= 15 + error))
            demod [i*8 + 7] =  ((abs(Q) >= 7 - error) and (abs (Q) <= 7 + error)) or ((abs(Q) >= 1 - error) and (abs (Q) <= 1 + error))  or ((abs(Q) >= 9 - error) and (abs (Q) <= 9 + error)) or ((abs(Q) >= 15 - error) and (abs (Q) <= 15 + error))
    
    return demod

x = np.arange(-4, 4 + 1) / np.sqrt (10)
x = x.reshape(-1, 1)
z = x + 1j * x 
BER_total = []
BER =[]
for i in range (50):
    payload = np.random.randint(0, 2, (1, A))
    bits = payload.flatten() 
    modulatedReturn = modulate (bits, "QAM16")


    plt.grid(True)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.xlabel('Real part (I)')
    plt.ylabel('Imaginary part (Q)')
    plt.title('QAM Constellation')
    plt.scatter (modulatedReturn.real, modulatedReturn.imag, s= 50, c = 'r', marker = 'o', label = 'Mod Signal')


    # mapping to REs
    reMap = np.zeros ((RB_allocated * 12, num_ofdm_symbol), dtype = complex)
    reMap[:, :] = modulatedReturn.reshape (RB_allocated * 12, num_ofdm_symbol)


    for SignalNoiseRatio in snrList:
        # OFDM modulation
        npad = (N - num_rb * 12) // 2
        remap_pad = (np.concatenate((np.zeros((npad, num_ofdm_symbol)), reMap, np.zeros((npad, num_ofdm_symbol)))))
        ifftout = np.fft.ifftshift(remap_pad) * np.sqrt(N)
        ifftout = np.fft.ifft(ifftout, axis=0)

        # Add CP
        signal = []
        for i in range(num_ofdm_symbol):
            if (i) % (7 * 2**mu) == 0:
                signal.append(ifftout[-int(CP0):, i])
                signal.append(ifftout[:, i])
            else:
                signal.append(ifftout[-int(CP1):, i])
                signal.append(ifftout[:, i])

        signal = np.concatenate(signal)
        TX_Signal = signal
        RX_Signal = channel (TX_Signal, SignalNoiseRatio)

        # Remove CP
        fftin = np.zeros((len(ifftout), num_ofdm_symbol), dtype=complex)
        offset = 0
        for i in range(num_ofdm_symbol):
            end = 0
            if (i) % (7 * 2**mu) == 0:
                offset += int(CP0)
            else:
                offset += int(CP1)
            for j in range (len(ifftout)):
                if (offset + j <= len (RX_Signal)):
                    fftin[j, i] = RX_Signal[offset+j]
            offset += len(ifftout)



        # OFDM de-modulation
        fftout = np.fft.fftshift(np.fft.fft(fftin, axis=0)) / np.sqrt(N)
        remap_rx = fftout[npad:npad+num_rb*12, :]

        # extract the REs
        returned = []
        for i in range (RB_allocated * 12):
            for j in range (Nsym_allocated):
                returned.append(remap_rx[Nrb_start + i, Nsym_start + j])


        returned = np.reshape(returned, (-1,))

        #getting the demod bits

        rx_demod_payload = demodulate (returned, "QAM16", 0.3)
        error_payload = rx_demod_payload - bits

        total_bits = 0
        for i in range (len (error_payload)):
            total_bits += abs(error_payload[i])

        BER.append((total_bits/len(bits)))

        RX_payload = rx_demod_payload.reshape (1,-1)
    
    
for i in range (len (snrList)):
    total_mean = np.mean(BER[i::len(snrList)])
    BER_total.append (total_mean)

plt.figure(figsize=(8,8))
plt.scatter (snrList, BER_total)
plt.xlabel('Signal-to-Noise Ratio (SNR)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Scatter Plot of SNR vs. BER')
plt.yscale ('log')
plt.grid()

    
error = modulatedReturn - returned
plt.figure()
plt.plot(np.abs(error), '.')
mse = np.mean(np.abs(error)**2) / np.mean(np.abs(modulatedReturn)**2)
mse_db = 10 * np.log10(mse)
print(f'MSE is {mse_db:.2f} dB')


plt.figure(figsize=(8,2))
plt.plot(abs(TX_Signal), label='TX signal')
plt.plot(abs(RX_Signal), label='RX signal')
plt.xlim (0, 80)
plt.legend(fontsize=10)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.grid(True)

plt.figure()
plt.scatter (returned.real, returned.imag, s = 100, marker = 'x',color = 'b', label = 'returned')
plt.scatter (modulatedReturn.real, modulatedReturn.imag, marker = 'o', color = 'r', label = 'initial')
plt.show()


