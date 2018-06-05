# SD 207: Statistical Machine Learning in practice
This repo contains the challenge done in the course 'SD 207: Statistical Machine Learning in practice' at Télécom ParisTech.

The challenge was to recognize accurately audio scenes from real life, with training.
We had lots of 30 seconds samples recorded inside bus, trains, at the beach, in restaurants...
We then wrote a system that could analyze the audio samples in order to recognize a new sample and predict the location of this new sample.

I used Python, with the following libraries:
- **pandas** to read CSV files
- **librosa** to analyze audio files, especially transform audio samples into real valued vectors
- **numpy** to perform computations on arrays (mean, standard deviation...)
- **sklearn** to construct classifiers

## Code structure
All the code is contained inside [one Jupyter notebook](https://github.com/LoicH/SD207/blob/master/challenge/loic_herbelot_challenge.ipynb).

The functions are all commented using **reST** style. 

**Example**
````python
def get_descr(path, n_coefs, n_vectors):
    """Returns a matrix description of an audio file.
  
	 :param path: The path to the audio file
	 :param n_coefs: The numbers of MFC coefficients
	 :param n_vectors: number of vectors that will represent the audio file
	  
	 :return: ndarray of shape (2 * n_coefs, n_vectors) 
	 where each column is the mean and standard deviation of MFCC sequence 
	 of multiple frames in the audio file."""
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coefs)
    # We have too many columns in [mfcc], we will group them into [X]
    X = np.zeros((2*n_coefs, n_vectors))
    # Numbers of samples we will take to make one vector
    p = mfcc.shape[1] // n_vectors 
    for i in range(n_vectors-1):
        # We collapse the [p] vectors into one single column in [X]
        X[:n_coefs,i]          = np.mean(mfcc[:,i*p:(i+1)*p], axis=1)
        X[n_coefs:2*n_coefs,i] = np.std(mfcc[:,i*p:(i+1)*p], axis=1)
    # Last vector:
    i += 1
    X[:n_coefs,i]          = np.mean(mfcc[:,i*p:], axis=1)
    X[n_coefs:2*n_coefs,i] = np.std(mfcc[:,i*p:], axis=1)
    return X
   ````
