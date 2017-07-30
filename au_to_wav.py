import sunau
import wave
 
def au2wav(au_file, wav_file):
    '''
        au_file: au filename
        wav_file: wave filename
        ex) au2wav('test.au', 'wav_file')
    '''
    
    au = sunau.open(au_file, 'r')
    wav = wave.open(wav_file, 'w')
 
    wav.setnchannels(au.getnchannels())
    wav.setsampwidth(au.getsampwidth())
    wav.setframerate(au.getframerate())
 
    wav.writeframes(au.readframes(au.getnframes()))
 
    wav.close()
    au.close()
au2wav(r'blues.00000.au',r'blue_wav.wav')
