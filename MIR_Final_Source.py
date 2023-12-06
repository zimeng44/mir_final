import mirdata
import librosa
import crepe
import numpy as np
import mir_eval
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import display, Audio

def tempo_match(ref_audio, input_audio):
    
    # Load the audio file
    ref_arr, sr = librosa.load(ref_audio.audio_path)
    in_arr, sr = librosa.load(in_audio.audio_path)

    # Tempo detector
    ref_tempo, beat_frames = librosa.beat.beat_track(y=ref_arr, sr=(sr*5))
    in_tempo, beat_frames = librosa.beat.beat_track(y=in_arr, sr=(sr*5))

    # Round up file's tempo to nearest 10 
    # round_tempo = int((round(tempo/10) * 10)) 

    # Calculate the time stretch factor needed to achieve the desired tempo
    stretch_factor = ref_tempo / in_tempo
    
    # Stretch the audio using the calculated stretch factor
    arr_stretched = librosa.effects.time_stretch(y = in_tempo, rate = stretch_factor)
    
    stretched_audio = Audio(arr_stretched, rate = sr)

def f0_slicer(audio):
    # cut audio into an array of audio frames by notes (f0), each frame has only one note(f0)
    # return an array of audio frames with estimated f0

    arr, sr = librosa.load(audio.audio_path)
    
    time, frequency, confidence, activation = crepe.predict(arr, sr, viterbi=False)

    frequency[confidence<0.3] = 0

    # plot_pitch(time, frequency, confidence, activation)
    # Create an array to store the sliced audio frames
    return_dict = {}

    i = 0

    # Iterate through note indices and slice audio accordingly
    for note in np.where(confidence > 0.8)[0]:

      start = time[note]

      if np.any(len(time) > note + 1):
        end = time[note+1]
        # print(end_time)
      else:
        end = start + 1

      # Convert time to sample indices
      start_sample = librosa.time_to_samples(start)
      # print(int(start_sample))
      end_sample = librosa.time_to_samples(end)
      # print(int(end_sample))

      # print(arr[start_sample:end_sample])

      # Slice audio and append to the array
      sliced_frames_arr = arr[start_sample:end_sample]
      time_code_arr = np.array([start_sample, end_sample])

      return_dict[i] = {'f0': frequency[note], 'time code': time_code_arr, 'audio': sliced_frames_arr}

      i+=1
    
    return sr, return_dict

def find_close_f0(input_audio_array, output_audio_array):
    # for each frame in input_audio_array, 
    # find frames in output_audio_array with a close f0 (no larger than 10 cent diviation).
    # return the found audio frames in output_audio_array as an new array
    # for each frame in input_audio_array, 
    # find frames in output_audio_array with a close f0 (no larger than 10 cent diviation).
    # return the found audio frames in output_audio_array as an new array
    # for frame_id, frame in input_audio_array.items():
    thres = 0.1
    hi = 2**(thres/12)
    lo = 1 / hi
    close_f0 ={}
    for id1, frame1 in input_audio_array.items():
      f1 = frame1['f0']
      fc = []
      if f1 > 0 :
        for id2, frame2 in output_audio_array.items():
          f2 = frame2['f0']
          if f2 > 0:
            if f2/f1 < hi and f2/f1 > lo: 
              fc.append(id2)
      close_f0[id1] = np.array(fc)
    return close_f0
    
def get_mfcc_stats(audio_array):
    # calculate MFCCs of each frame in the array, 
    # and store mean and std of each frame in a new array  or dict
    # return a array of mfcc means and std
    feature_matrix = {}
    for k,v in audio_array.items():
      
      n_fft = 2048
      sr = 22050
      hop_length = 512
      n_mels = 128
      n_mfcc = 20
      mel_spectrogram = librosa.feature.melspectrogram(y=v['audio'], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
      mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc)
      mfcc = mfcc[1:n_mfcc]
      features_mean = np.mean(mfcc, axis=-1)
      feature_matrix[k] = features_mean
    return feature_matrix
    
def find_close_mfcc(input_audio_array, output_audio_array):
    # for each frame in input_audio_array, 
    # we can find a few frames in output_audio_array that have close f0 (with 'find_close_f0()')
    # In these close f0 frames, find the one with the closet mfcc to the frame in input_audio_array
    #return an array of the frames with the closest MFCCs
    pass

def resynth(input_audio_array, input_mfcc_array, output_audio_array, output_mfcc_array):
    #put the frames in output_audio_array in the mapped order with adjusted length. 
    #return the resynthesized audio
    pass
