"""features.mfcc() - Mel Frequency Cepstral Coefficients
features.fbank() - Filterbank Energies
features.logfbank() - Log Filterbank Energies
features.ssc() - Spectral Subband Centroids
"""
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
coun=0
kkk=7
while (kkk==7):
    (rate,sig) = wav.read("blues.0000"+str(kkk)+".wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate, winlen=0.03, winstep=0.03)

    #print fbank_feat[0]
    normalised=[]
    for i in fbank_feat:
        sublist=[]
        for j in i:
            sublist.append(int(round(j/22*7)))
        normalised.append(sublist)
    with open("blue.txt", "a") as myfile:
        for i in normalised:
            print i
            for j in i:
                myfile.write(str(j))
                coun+=1
    kkk=kkk+1

'''
kkk=10
while (kkk<100):
    (rate,sig) = wav.read("blues.000"+str(kkk)+".wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate, winlen=0.03, winstep=0.03)

    #print fbank_feat[0]
    normalised=[]
    for i in fbank_feat:
        sublist=[]
        for j in i:
            sublist.append(int(round(j/22*7)))
        normalised.append(sublist)
    with open("blue.txt", "a") as myfile:
        for i in normalised:
            print i
            for j in i:
                myfile.write(str(j))
                coun+=1
    kkk=kkk+1
print coun
"""
i=0
C:\Users\MegK\Desktop\wavs\ble\blues.0000"+
"""
'''
