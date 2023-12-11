
import librosa
import crepe
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import display, Audio

import math

def tempo_match(ref_audio, match_audio):
    
    # Load the audio file
    ref_arr, sr_ref = ref_audio
    match_arr, sr_match = match_audio

    # Tempo detector
    ref_tempo, beat_frames = librosa.beat.beat_track(y=ref_arr, sr=sr_ref)
    match_tempo, beat_frames = librosa.beat.beat_track(y=match_arr, sr=sr_match)

    # print(ref_tempo, match_tempo)

    # Round up file's tempo to nearest 10 
    # round_tempo = int((round(tempo/10) * 10)) 

    # Calculate the time stretch factor needed to achieve the desired tempo
    stretch_factor =  match_tempo/ref_tempo
    
    # Stretch the audio using the calculated stretch factor
    arr_stretched = [librosa.effects.time_stretch(y = match_arr, rate = stretch_factor), sr_match]

    return arr_stretched
    
    # stretched_audio = Audio(arr_stretched, rate = sr)


def f0_slicer(audio):
    # cut audio into an array of audio frames by notes (f0), each frame has only one note(f0)
    # return an array of audio frames with estimated f0

    # arr, sr = librosa.load(audio.audio_path)
    arr = audio[0]
    sr = audio[1]
    
    time, frequency, confidence, activation = crepe.predict(arr, sr, viterbi=True)

    frequency[confidence<0.3] = 0

    # Create an array to store the sliced audio frames
    slices = {}

    # # Iterate through note indices and slice audio accordingly
   
    cents = 20
    hi = 2**(cents/1200)
    lo = 1 / hi

    slice_id = 0
    i = 0

    while i < (len(frequency) - 1):
      note_times = [librosa.time_to_samples(time[i], sr = sr), 0]
      mean_freq = [frequency[i], ]

      for j in np.arange(i + 1, len(time)): # j starts from the next element to i
          
          if j >= len(frequency)-1: #if the loop has reach the end of thefrequency  array, quit looping
            note_times[1] = librosa.time_to_samples(time[j], sr = sr)
            mean_freq.append(frequency[j])

            i = j
            # print("1", i, j)
            break
          
          if frequency[i] == 0:           # if frequency[i] is equal to 0
             
             if frequency[j] == 0:        # if frequency[i] and frequency[j] are both equal to 0, keep looping

               #  print("2", i, j)
                continue
             else:                        #if frequency[i] is 0 but frequency[j] is not 0, we found the end of the note_time
                if j-1 == i:
                   continue
                note_times[1] = librosa.time_to_samples(time[j-1], sr = sr)
                mean_freq.append(frequency[j])

                i = j                     # in next iteration, i stats at the j position
               #  print("3", i, j)
                break
             
          else: # if frequency[i] is not equal to 0
             
             if frequency[j] == 0:        #if frequency[i] is not 0 but frequency[j] is  0, we found the end of the note_time
                if j-1 == i:
                   continue
                note_times[1] = librosa.time_to_samples(time[j-1], sr = sr)
                mean_freq.append(frequency[j])

                i = j                     # in next iteration, i stats at the j position
               #  print("4", i, j)
                break
             
             if frequency[j]  / frequency[i] < hi and frequency[i+1]  / frequency[i] > lo:  #if frequency[i] is not 0 and frquency[j] is a close f0

               #  print("5", i, j)
                i = j
                mean_freq.append(frequency[j])
                continue                 # we need to find where the note ends
             else:                       #if frequency[i] is not 0 and frquency[j] is not a close f0, we found the end of the note
                if j-1 == i:
                   continue
                note_times[1] = librosa.time_to_samples(time[j-1], sr = sr) 
                mean_freq.append(frequency[j])
               
                i = j                    # in next iteration, i stats at the j position
               #  print("6", i, j)
                break
             
      # print("7", i)
      # print(i, note_times)
      audio = np.array([arr[note_times[0]:note_times[1]], sr], dtype=object)
      f0_mean = np.mean(mean_freq)
      # print(np.array(audio))
      slices[slice_id] = {'f0': f0_mean, 'time_code': np.array(note_times), 'audio': audio}
      slice_id += 1

   #  for i in slices:
   #     if len(slices[i]['audio'][0]) == 0:
   #        print(i)
    
    return slices

def find_close_f0(ref_slices, match_slices):
    # for each frame in input_audio_array, 
    # find frames in output_audio_array with a close f0 (no larger than 10 cent diviation).
    # return the found audio frames in output_audio_array as an new array
    # for each frame in input_audio_array, 
    # find frames in output_audio_array with a close f0 (no larger than 10 cent diviation).
    # return the found audio frames in output_audio_array as an new array
    # for frame_id, frame in input_audio_array.items():
    cents = 10
    hi = 2**(cents/1200)
    lo = 1 / hi
    close_f0 = {}

    for id_ref, frame_ref in ref_slices.items():

      f0_ref = frame_ref['f0']
      matched_ids = []

      if f0_ref > 0 :
        for id_match, frame_match in match_slices.items():
          
          f0_match = frame_match['f0']

          if f0_match > 0:
            for octave in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]:
              if (f0_match*octave) / f0_ref < hi and (f0_match*octave) / f0_ref > lo: 
                  matched_ids.append([id_match, octave])
                  # print(f0_ref,f0_match)
                  break
        if len(matched_ids) == 0:
           matched_ids = np.array([[-1, 0]])

        close_f0[id_ref] = np.array(matched_ids)
        
      else:
         
         close_f0[id_ref] = np.array([[-1, 0]])
         # print(close_f0[id_ref])

    return close_f0
    
def get_mfcc_stats(slices):
    # calculate MFCCs of each frame in the array, 
    # and store mean and std of each frame in a new array  or dict
    # return a array of mfcc means and std
    feature_matrix = {}
    for k,v in slices.items():
      
      n_fft = 128
      sr = v['audio'][1]
      hop_length = 512
      n_mels = 16
      n_mfcc = 20

      mel_spectrogram = librosa.feature.melspectrogram(y = v['audio'][0], sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
      mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc)
      mfcc = mfcc[1:n_mfcc]
      features_mean = np.mean(mfcc, axis=-1)
      feature_matrix[k] = features_mean

    return feature_matrix
    
def find_close_mfcc(close_f0, ref_mfcc, match_mfcc):
    # for each frame in input_audio_array, 
    # we can find a few frames in output_audio_array that have close f0 (with 'find_close_f0()')
    # In these close f0 frames, find the one with the closet mfcc to the frame in input_audio_array
    #return an array of the frames with the closest MFCCs
    # pass

    close_mfcc = {}

    for id, all_matched_f0 in close_f0.items():

      if all_matched_f0[0][0] != -1: # if there are matched frames for the ref_id in ref_slices
         
         cloest_distance = np.sum(match_mfcc[all_matched_f0[0][0]]) - np.sum(ref_mfcc[id])
         close_mfcc[id] = all_matched_f0[0]

         for f0_matched_id in all_matched_f0: # find the closest mfcc match for the ref_id
            
            # print(match_mfcc[f0_matched_id[0]])
            distance = np.sum(match_mfcc[f0_matched_id[0]]) - np.sum(ref_mfcc[id])
            
            if distance < cloest_distance:
               cloest_distance = distance
               close_mfcc[id] = np.array(f0_matched_id)
      else:
         
         close_mfcc[id] = np.array([-1, 0])

    return close_mfcc

def resynth(ref_slices, match_slices, close_mfcc):
    #put the frames in output_audio_array in the mapped order with adjusted length. 
    #return the resynthesized audio
    # pass
    output_audio = np.array([])
    
    for frame_id, ref_slice in ref_slices.items():

      if close_mfcc[frame_id][0] == -1:

         # silent_slice = ref_slice[ref_slice['time_code'][0]:ref_slice['time_code'][1]]
         output_audio = np.concatenate((output_audio, ref_slice['audio'][0]))
        #  print("found zeros", silent_slice)

      else:

         ref_frame_len = len(ref_slice['audio'])
         match_frame_len = len(match_slices[close_mfcc[frame_id][0]]['audio'])
         time_stretch_rate = 1

         if match_frame_len > 0 and ref_frame_len>0:
            time_stretch_rate = ref_frame_len/match_frame_len

         matched_audio  = match_slices[close_mfcc[frame_id][0]]['audio'][0]
         
         matched_audio = librosa.effects.time_stretch(y = matched_audio, rate = 1/time_stretch_rate)

         sr = match_slices[close_mfcc[frame_id][0]]['audio'][1]

         octave = close_mfcc[frame_id][1]

         if octave > 1:
            matched_audio = librosa.effects.pitch_shift(y = matched_audio, sr = sr, n_steps = (octave-1) * 12, )
         elif 0 < octave < 1:
            matched_audio = librosa.effects.pitch_shift(y = matched_audio, sr = sr, n_steps = math.log2(octave) * 12 )
            
         output_audio = np.concatenate((output_audio, matched_audio))

    return output_audio
