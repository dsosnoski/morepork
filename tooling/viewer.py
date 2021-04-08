import librosa.display
import librosa.feature as feature
import matplotlib.pyplot as plt
import numpy as np
import pickle

nummels = 60
freqmin = 600
freqmax = 1200
slicesPerSecond = 20
secondsPerSample = 10
slicesPerSample = slicesPerSecond * secondsPerSample


def scale(segments):
    for key in segments.keys():
        segment = segments[key]
        segment = librosa.power_to_db(segment, ref=np.max)
        segments[key] = segment

def save_spectfig(spec, name, title):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(np.squeeze(spec), hop_length=1102, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Samples in a segment')
    plt.savefig(f'/tmp/{name}.png', bbox_inches=None, pad_inches=0)
    plt.close()
    #plt.show(block=True)

def loadSample(path):
    frames, sr = librosa.load(path, sr=None)

    # generate spectrogram
    nfft = int(sr / (slicesPerSecond * .5))
    specgram = feature.melspectrogram(
        y=frames,
        sr=sr,
        n_fft=nfft,
        hop_length=int(nfft / 2),
        n_mels=nummels,
        fmin=freqmin,
        fmax=freqmax)

    return (frames, sr, np.array(specgram))

segments_path = '/mnt/tmp/morepork/training-samples-66-570-1260.pk'
with open(segments_path, 'rb') as f:
    positive_segments = pickle.load(f)
    negative_segments = pickle.load(f)
scale(positive_segments)
scale(negative_segments)

for k in positive_segments:
    save_spectfig(positive_segments[k], k, f'{k}')