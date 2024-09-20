# Mod-Demod
**Aim of Project:** </br>
o	Generated constellations for QAM schemes (4-QAM, 16-QAM, 64-QAM, 256-QAM) from 100,000 bits and modulated OFDM symbols using Inverse Fast Fourier Transformations for various QAM schemes. </br>
o	Added Additive White Gaussian Noise (AWGN) to the modulated signals to simulate real-world wireless channel conditions. </br>
o	Demodulated noisy signals with hard & soft decision techniques into bits, comparing error rate performance and analyzing trade-offs between each different signal-to-noise ratio </br>
o	Calculated the Log-Likelihood Ratios (LLRs) for each demodulated bit through Max-Log and Log Map. </br>
o Developed & evaluated "LLRNet", a neural network model for LLR estimation, comparing its performance with Max-Log and Log Map, and plotting Bit Error Rate (BER) against Signal-to-Noise Ratio </br>
</br>
Libraries Required: matplotlib, numpy, keras, tensorflow </br>
Instruction: download required libraries and run on VSCode or Jupyter Notebook (or any other) 
