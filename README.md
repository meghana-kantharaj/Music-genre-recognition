README - Music Genre Recognition

Files included

.py files
au_to_wav
generate_feature_vectors
training testing

input files are audio files of a .au extension

au_to_wav.py converts input .au file to .wav file. The input names are mentioned in the calling function in the script.

generate_feature_vectors.py converts the audio wav to features.
The audio file signal rate and directory are read into the program. These are used to convert the signal into mel filterbank energies with standard window lengths 0.03 and window step 0.03 ( more here - http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html). Then, the mel filterbank energies are normalized and written into a file.

trainingtesting.py considers 4 HMM models and trains them using initialized random weights and normalized data each for each HMM model. It then calculates and prints the confusion matrix.

mymodel.json is a random initialization for the HMM parameters

mymodel.json is the best fit parameters for the HMM classifier

blues00000 is a .au file sample to input into au_to_wav.py

blues00007 is a program converted .wav file, output by au_to_wav.py

class.txt and rock.txt are the normalized mel filterbank energies for classical and rock music genres.
