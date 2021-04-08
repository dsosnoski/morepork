import math
import numpy as np
#import pandas as pd
import librosa
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import pyaudio
import sqlite3
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.widgets import Button
from scipy.signal import butter, sosfilt

training_data = False

freqmin = 600
freqmax = 1200
bands = 60
persec = 20
window_size = 3.0
zoom_seconds = 8.0
highlight_box_color = (0.2, 0.1, 0.0, 0.1)
selection_box_color = (0.2, 0.2, 0.2, 0.2)
if training_data:
    segment_base_path = '/mnt/tmp/morepork/training-recordings'
else:
    segment_base_path = '/home/dennis/projects/morepork/march-test'
db_path = '/home/dennis/projects/timserver/audio_analysis_chris.db'
db_connection = None
try:
    db_connection = sqlite3.connect(db_path)
except sqlite3.Error as e:
    print(e)

code_color = (0.0, 0.0, 1.0, 0.3)
code_color_alt = (0.0, 0.3, 0.3, 0.2)
code_to_hatch_and_color = { 'morepork': ('//', code_color), 'maybepork': ('xx', code_color),
                            'other': ('\\\\', code_color), 'duck': ('o', code_color), 'dove': ('*', code_color),
                            'tim_morepork': ('//', code_color_alt), 'tim_otherpork': ('xx', code_color_alt) }
replace_tags = { 'morepork_more-pork': 'tim_morepork', 'maybe_morepork_more-pork': 'tim_otherpork',
                 'morepork_more-pork_part': 'tim_otherpork', 'morepork_croaking': 'tim_otherpork'}

#https://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
#https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively


class RecordingStore:

    def __init__(self, base_path, false_positives, false_negatives, db_connection):
        self.segment_base_path = base_path
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.db_connection = db_connection
        self.highlights = {}

    def get_recording(self, recording_id):
        path = f'{self.segment_base_path}/{recording_id}.m4a'
        return librosa.load(path, sr=None)

    def get_highlights(self, recording_id):
        if recording_id in self.highlights:
            return self.highlights.get(recording_id)
        else:
            try:
                sql = ''' SELECT * FROM chris_tags
                          WHERE recording_id = ?
                          ORDER BY start_time  '''
                cur = db_connection.cursor()
                cur.execute(sql, [recording_id])
                tags = cur.fetchall()
            except Exception as e:
                print(e, '\n')
                print('Failed to get chris tags\n')
            highlights = []
            for tuple in tags:
                recording_id = tuple[1]
                start_time = tuple[3] - 0.5
                end_time = tuple[3] + 0.5
                highlight = (tuple[2], recording_id, start_time, end_time)
                highlights.append(highlight)
            self.highlights[recording_id] = highlights
            print(f'retrieved chris tags {highlights}')
            return highlights

    def get_description(self, recording_id):
        try:
            sql = ''' SELECT device_name, device_super_name, recordingDateTimeNZ FROM recordings
                      WHERE recordings.recording_id = ?  '''
            cur = self.db_connection.cursor()
            cur.execute(sql, [int(recording_id)])
            result = cur.fetchone()
            return f'Recording {recording_id} from {result[0]} at {result[1]} time {result[2]} with {self.false_positives[recording_id]} false positives and {self.false_negatives[recording_id]} false negatives'
        except Exception as e:
            print(e, '\n')
            print('Failed to get description ' + str(recording_id), '\n')


class TagHandler:

    tag_cache = {}
    db_connection = None

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_tags(self, recording_id):
        if recording_id in self.tag_cache:
            return self.tag_cache[recording_id]
        else:
            try:
                cur = self.db_connection.cursor()
                sql = ''' SELECT start_time_seconds, finish_time_seconds, what FROM training_validation_test_data
                          WHERE recording_id = ?  '''
                cur.execute(sql, [int(recording_id)])
                tags = list(cur.fetchall())
                self.tag_cache[recording_id] = tags
                print(f'retrieved tags {tags}')
                return tags
            except Exception as e:
                print(e, '\n')
                print('Failed to get tags ' + str(recording_id), '\n')

    def remove_tag(self, recording_id, start, end):
        if recording_id in self.tag_cache:
            tags = self.tag_cache[recording_id]
            for i in tags:
                if i[0] == start and i[1] == end:
                    tags.remove(i)
                    try:
                        sql = ''' DELETE FROM dennis_tags
                                  WHERE dennis_tags.recording_id = ? AND dennis_tags.start_time_seconds = ? AND dennis_tags.finish_time_seconds = ?  '''
                        cur = self.db_connection.cursor()
                        cur.execute(sql, (int(recording_id), start, end))
                        self.db_connection.commit()
                    except Exception as e:
                        print(e, '\n')
                        print('Failed to remove tag ' + str(recording_id), '\n')

    def add_tag(self, recording_id, start, end, code):
        if recording_id in self.tag_cache:
            self.tag_cache[recording_id].append((start, end, code))
        else:
            self.tag_cache[recording_id] = [(start, end, code)]
        try:
            sql = ''' INSERT INTO dennis_tags(recording_id, start_time_seconds, finish_time_seconds, what)
                      VALUES(?, ?, ?, ?)  '''
            cur = self.db_connection.cursor()
            cur.execute(sql, (int(recording_id), start, end, code))
            self.db_connection.commit()
        except Exception as e:
            print(e, '\n')
            print('Failed to add tag ' + str(recording_id), '\n')


class SpectrogramViewer:

    figure, (full_spectrum_ax, zoom_spectrum_ax) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    figure.subplots_adjust(bottom=0.25)
    adjusted_spectrogram = None
    select_rect = None
    full_image = None
    zoom_image = None

    # define axes areas and buttons
    last_recording_ax = figure.add_axes([0.01, 0.10, 0.12, 0.03])
    last_recording_button = Button(last_recording_ax, '<<')
    last_highlight_ax = figure.add_axes([0.14, 0.10, 0.12, 0.03])
    last_highlight_button = Button(last_highlight_ax, '<')
    playbuttonax = figure.add_axes([0.28, 0.10, 0.14, 0.03])
    playbutton = Button(playbuttonax, 'Play')
    play_full_buttonax = figure.add_axes([0.43, 0.10, 0.14, 0.03])
    play_full_button = Button(play_full_buttonax, 'Play Adjusted')
    play_filtered_buttonax = figure.add_axes([0.58, 0.10, 0.14, 0.03])
    play_filtered_button = Button(play_filtered_buttonax, 'Play Filtered')
    next_highlight_button_ax = figure.add_axes([0.74, 0.10, 0.12, 0.03])
    next_highlight_button = Button(next_highlight_button_ax, '>')
    next_recording_button_ax = figure.add_axes([0.87, 0.10, 0.12, 0.03])
    next_recording_button = Button(next_recording_button_ax, '>>')
    clear_button_ax = figure.add_axes([0.10, 0.05, 0.11, 0.03])
    clear_button = Button(clear_button_ax, 'Clear')
    morepork_button_ax = figure.add_axes([0.26, 0.05, 0.11, 0.03])
    morepork_button = Button(morepork_button_ax, 'Morepork')
    dog_button_ax = figure.add_axes([0.39, 0.05, 0.11, 0.03])
    dog_button = Button(dog_button_ax, 'Dog')
    duck_button_ax = figure.add_axes([0.52, 0.05, 0.11, 0.03])
    duck_button = Button(duck_button_ax, 'Duck')
    dove_button_ax = figure.add_axes([0.65, 0.05, 0.11, 0.03])
    dove_button = Button(dove_button_ax, 'Dove')
    notpork_button_ax = figure.add_axes([0.78, 0.05, 0.11, 0.03])
    notpork_button = Button(notpork_button_ax, 'Other')

    # next row of buttons: "<<", "Morepork", "Notport", "Maybepork", ">>"
    mouse_down = False
    mouse_start = None
    mouse_end = None
    select_rect = Rectangle((0,0), 0, bands, fill=False)
    select_rect.set_color(selection_box_color)
    select_rect.set_linewidth(2.0)
    zoom_spectrum_ax.add_patch(select_rect)
    highlight_rect = Rectangle((0,0), 0, bands, fill=False)
    highlight_rect.set_color(highlight_box_color)
    highlight_rect.set_linewidth(2.0)
    zoom_spectrum_ax.add_patch(highlight_rect)
    zoom_selection_rect = Rectangle((0,0), 0, bands, fill=False, color='k')
    full_spectrum_ax.add_patch(zoom_selection_rect)
    full_highlight_texts = []
    full_highlight_patches = []
    zoom_highlight_patches = []
    full_label_patches = []
    zoom_label_patches = []
    return_value = None

    legend_elements = [Patch(facecolor='orange', edgecolor='b', hatch='//', label='Morepork'),
                       Patch(facecolor='orange', edgecolor='b', hatch='xx', label='Maybepork'),
                       Patch(facecolor='orange', edgecolor='b', hatch='\\\\', label='Other'),
                       Patch(facecolor='orange', edgecolor='b', hatch='oo', label='Duck'),
                       Patch(facecolor='orange', edgecolor='b', hatch='**', label='Dove')]
    full_spectrum_ax.legend(handles=legend_elements)

    pa = pyaudio.PyAudio()
    audio_stream = None


    def selection_time_span(self):
        if self.mouse_start != None and self.mouse_end != None:
            start = (min(self.mouse_start, self.mouse_end) + self.zoom_start_column) / persec
            end = (max(self.mouse_start, self.mouse_end) + self.zoom_start_column) / persec
            return (start, end)
        else:
            return (0, 0)

    def play_selection(self, frames, sr, adjust):
        ratio = sr / persec
        if self.mouse_start and self.mouse_end:
            start = int((min(self.mouse_start, self.mouse_end) + self.zoom_start_column) * ratio)
            end = int((max(self.mouse_start, self.mouse_end) + self.zoom_start_column) * ratio)
            if adjust:
                sound = self.adjust_amplitude(frames[start:end])
            else:
                sound = (frames[start:end] * 32767.0).astype('int16')
        elif adjust:
            sound = self.adjust_amplitude(frames[int(self.zoom_start_column * ratio):int(self.zoom_end_column * ratio)])
        else:
            sound = (frames[int(self.zoom_start_column * ratio):int(self.zoom_end_column * ratio)] * 32767.0).astype('int16')
        self.audio_stream.write(sound.tobytes())
        self.figure.canvas.draw_idle()
        #self.figure.canvas.flush_events()

    def play(self, event):
        self.play_selection(self.recording_frames, self.recording_rate, False)

    def play_full(self, event):
        self.play_selection(self.recording_frames, self.recording_rate, True)

    def play_filtered(self, event):
        filtered = self.butter_bandpass_filter(self.recording_frames, freqmin, freqmax, self.recording_rate, order=7)
        self.play_selection(filtered, self.recording_rate, True)

    def last_recording(self, event):
        self.set_recording(self.current_position-1)

    def next_recording(self, event):
        self.set_recording(self.current_position+1)

    def last_highlight(self, event):
        self.set_highlight(self.current_highlight-1)
        self.figure.canvas.draw_idle()
        #self.figure.canvas.flush_events()

    def next_highlight(self, event):
        self.set_highlight(self.current_highlight+1)
        self.figure.canvas.draw_idle()
        #self.figure.canvas.flush_events()

    def on_press(self, event):
        #print(f'on_press event.xdata={event.xdata}, event.ydata={event.ydata}')
        if event.xdata and event.inaxes == self.zoom_spectrum_ax:
            self.mouse_start = event.xdata
            self.mouse_down = True
            self.select_rect.set_width(0)
            self.select_rect.set_height(self.adjusted_spectrogram.shape[1])
            self.select_rect.set_xy((self.mouse_start, 0))
            self.select_rect.fill = False
            self.select_rect.set_linewidth(2.0)
        elif event.xdata and event.inaxes == self.full_spectrum_ax:
            self.set_zoom(event.xdata / persec)
            self.set_zoom_tags()
        else:
            self.mouse_down = False
        self.figure.canvas.draw_idle()
        #self.figure.canvas.flush_events()

    def on_motion(self, event):
        #print(f'on_motion event.xdata={event.xdata}, event.ydata={event.ydata}')
        if self.mouse_down and event.xdata and event.inaxes == self.zoom_spectrum_ax:
            self.mouse_end = event.xdata
            self.select_rect.set_width(abs(self.mouse_start - self.mouse_end))
            self.select_rect.set_height(self.adjusted_spectrogram.shape[1])
            self.select_rect.set_xy((min(self.mouse_start, self.mouse_end), 0))
            self.figure.canvas.draw_idle()
            #self.figure.canvas.flush_events()

    def draw_selection(self):
        self.select_rect.set_width(abs(self.mouse_start - self.mouse_end))
        self.select_rect.set_height(self.adjusted_spectrogram.shape[1])
        self.select_rect.set_xy((min(self.mouse_start, self.mouse_end), 0))
        self.select_rect.fill = True

    def on_release(self, event):
        #print(f'on_release event.xdata={event.xdata}, event.ydata={event.ydata}')
        if self.mouse_down and event.inaxes == self.zoom_spectrum_ax:
            self.mouse_end = event.xdata
            self.mouse_down = False
            self.draw_selection()
            self.figure.canvas.draw_idle()
            #self.figure.canvas.flush_events()
        else:
            self.mouse_down = False

    def add_tag(self, tag):
        start, end = self.selection_time_span()
        if end > start:
            self.tag_handler.add_tag(self.recording_id, start, end, tag)
            self.tags = self.tag_handler.get_tags(self.recording_id)
            self.refresh_full_tags_highlights()

    def tag_morepork(self, event):
        self.add_tag('morepork')

    def tag_dog(self, event):
        self.add_tag('dog')

    def tag_duck(self, event):
        self.add_tag('duck')

    def tag_dove(self, event):
        self.add_tag('dove')

    def tag_notpork(self, event):
        self.add_tag('other')

    def clear_tag(self, event):
        start, end = self.selection_time_span()
        cleared = False
        for tag in self.tags:
            if tag[0] <= end and tag[1] >= start:
                self.tag_handler.remove_tag(self.recording_id, tag[0], tag[1])
                self.tags = self.tag_handler.get_tags(self.recording_id)
                cleared = True
        if cleared:
            self.refresh_full_tags_highlights()

    def __init__(self, tag_handler, data_store, recordings, initial):
        self.tag_handler = tag_handler
        self.data_store = data_store
        self.recordings_list = recordings
        self.playbutton.on_clicked(self.play)
        self.play_full_button.on_clicked(self.play_full)
        self.play_filtered_button.on_clicked(self.play_filtered)
        self.last_recording_button.on_clicked(self.last_recording)
        self.next_recording_button.on_clicked(self.next_recording)
        self.last_highlight_button.on_clicked(self.last_highlight)
        self.next_highlight_button.on_clicked(self.next_highlight)
        self.clear_button.on_clicked(self.clear_tag)
        self.morepork_button.on_clicked(self.tag_morepork)
        self.dog_button.on_clicked(self.tag_dog)
        self.duck_button.on_clicked(self.tag_duck)
        self.dove_button.on_clicked(self.tag_dove)
        self.notpork_button.on_clicked(self.tag_notpork)
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.set_recording(initial)
        plt.show()

    def plot_spectrogram(self, spec, offset, image,  axis):
        axis.cla()
        image = axis.imshow(spec, cmap='magma', origin='lower', aspect='auto', interpolation='hamming')

        numseconds = int((spec.shape[1] + 1) / persec)
        position_adjust = 1 / persec
        seconds_scale = 1
        freq_mult = 10
        if numseconds > 10:
            seconds_scale = 5
            freq_mult = 20

        ## create y axis
        freqstep = int((freqmax - freqmin) / spec.shape[0])
        ylabels = [i for i in range(freqmin, freqmax + 1, freqstep * freq_mult)]
        yticks = [i * freq_mult - 0.5 for i in range(0, len(ylabels))]
        axis.set_yticks(yticks)
        axis.set_yticklabels(ylabels)
        axis.set_ylabel("Frequency (Hz)")

        ## create x axis
        start_second = math.ceil(offset / seconds_scale) * seconds_scale
        first_tick = (start_second - offset) * persec - position_adjust
        num_ticks = int((spec.shape[1] / persec + offset - start_second) / seconds_scale) + 1
        xticks = [i * persec * seconds_scale + first_tick for i in range(0, num_ticks)]
        print(f'spectrogram length {spec.shape[1]} ({spec.shape[1]/persec} seconds), xticks={xticks}')
        xlabels = [f'{i*seconds_scale+start_second}.0' for i in range(len(xticks))]
        axis.set_xticks(xticks)
        axis.set_xticklabels(xlabels)
        axis.set_xlabel("Time (sec)")
        return image

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
        return sosfilt(sos, data)

    def adjust_amplitude(self, frames):
        adjust = max(abs(frames)) * 1.2
        return (frames * 32767.0 / adjust).astype('int16')

    def time_to_column(self, time):
        return int(time * persec)

    def time_to_upper_column(self, time):
        return math.ceil(time * persec)

    def time_to_zoom(self, time):
        return self.time_to_column(time) - self.zoom_start_column

    def set_zoom(self, instant):
        start_index = max(0, int((instant - zoom_seconds / 2) * persec))
        end_index = start_index + int(zoom_seconds*persec) + 1
        if end_index > self.adjusted_spectrogram.shape[1]:
            end_index = self.adjusted_spectrogram.shape[1]
            start_index = end_index - int(zoom_seconds*persec) - 1
        slice = self.adjusted_spectrogram[:, start_index:end_index]
        self.select_rect.set_width(0)
        self.select_rect.fill = False
        self.highlight_rect.set_width(0)
        self.highlight_rect.fill = False
        self.zoom_start_column = start_index
        self.zoom_end_column = start_index + self.time_to_column(zoom_seconds)
        self.zoom_image = self.plot_spectrogram(slice, start_index / persec, self.zoom_image, self.zoom_spectrum_ax)
        self.mouse_end = None
        self.mouse_start = None
        self.zoom_spectrum_ax.patches = []
        self.zoom_selection_rect.set_x(start_index)
        self.zoom_selection_rect.set_width(end_index - start_index)
        self.zoom_spectrum_ax.add_patch(self.select_rect)
        self.zoom_spectrum_ax.add_patch(self.highlight_rect)
        self.set_zoom_tags()
        self.conn_patch_left = ConnectionPatch(xyA=(start_index,0), xyB=(0,bands), coordsA='data', coordsB='data', axesA=self.full_spectrum_ax, axesB=self.zoom_spectrum_ax)
        self.conn_patch_left.set_color([0,0,0])
        self.conn_patch_left.set_linewidth(2)
        self.full_spectrum_ax.add_artist(self.conn_patch_left)
        self.figure.canvas.draw()

    def what_hatch_and_color(self, code):
        if code in replace_tags:
            code = replace_tags[code]
        return code_to_hatch_and_color.get(code, ('\\\\', code_color_alt))

    def set_zoom_tags(self):
        for tag in self.tags:
            left = self.time_to_column(tag[0])
            right = self.time_to_column(tag[1])
            if left < self.zoom_end_column and right > self.zoom_start_column:
                zoom_left = max(0, left - self.zoom_start_column)
                zoom_right = min(self.zoom_end_column - self.zoom_start_column, right - self.zoom_start_column)
                hatch, color = self.what_hatch_and_color(tag[2])
                rect = Rectangle((zoom_left, 0), zoom_right - zoom_left, bands, hatch=hatch, edgecolor=color, facecolor=None, fill=False)
                rect.set_linewidth(2.0)
                self.zoom_spectrum_ax.add_patch(rect)

    def set_highlight(self, index):
        self.current_highlight = index
        if len(self.highlights) > 0:
            highlight = self.highlights[index]
            self.set_zoom((highlight[2]+highlight[3]) / 2)
            highstart = self.time_to_zoom(highlight[2])
            highend = self.time_to_zoom(highlight[3])
            self.highlight_rect.set_width(highend-highstart)
            self.highlight_rect.set_xy((highstart, 0))
            self.highlight_rect.fill = True
            self.last_highlight_button.set_active(index > 1)
            self.next_highlight_button.set_active(index < len(self.highlights) - 1)
            self.mouse_start = highstart
            self.mouse_end = highend
            self.draw_selection()
        else:
            self.set_zoom(0)
            self.last_highlight_button.set_active(False)
            self.next_highlight_button.set_active(False)
        self.set_zoom_tags()
        # self.figure.canvas.draw()

    def set_full_highlights(self):
        for i in range(len(self.full_highlight_texts), len(self.highlights)):
            self.full_highlight_texts.append(self.full_spectrum_ax.text(0, 0, '', color='white'))
            rect = Rectangle((0, 0), 0, 0, fill=False)
            rect.set_color(highlight_box_color)
            rect.set_linewidth(2.0)
            self.full_highlight_patches.append(rect)
        vert_base = bands * 5 / 6
        last_left = -100
        for (highlight, text, patch) in zip(self.highlights, self.full_highlight_texts, self.full_highlight_patches):
            left = self.time_to_column(highlight[2])
            width = self.time_to_column(highlight[3]) - left + 1
            patch.set_bounds((left, 0, width, bands))
            patch.fill = True
            self.full_spectrum_ax.add_patch(patch)
            value = highlight[0]
            text.set_text(str(value))
            if (left - last_left > persec * 4) or vert_base < bands / 3:
                vert_base = bands * 5 / 6
            else:
                vert_base -= 8
            text.set_position((left, vert_base))
            last_left = left
            self.full_spectrum_ax.texts.append(text)

    def set_full_tags(self):
        for i in range(len(self.full_label_patches), len(self.tags)):
            rect = Rectangle((0, 0), 0, 0, facecolor=None, fill=False)
            rect.set_linewidth(2.0)
            self.full_label_patches.append(rect)
        for (tag, patch) in zip(self.tags, self.full_label_patches):
            left = self.time_to_column(tag[0])
            limit = min(self.time_to_column(tag[1]) + 1, self.adjusted_spectrogram.shape[1])
            width = limit - left
            hatch, color = self.what_hatch_and_color(tag[2])
            patch.set_bounds((left, 0, width, bands))
            patch.set_hatch(hatch)
            patch.set_edgecolor = color
            self.full_spectrum_ax.add_patch(patch)

    def refresh_full_tags_highlights(self):
        self.full_spectrum_ax.patches = []
        self.full_spectrum_ax.texts = []
        self.full_spectrum_ax.add_patch(self.zoom_selection_rect)
        self.set_full_highlights()
        self.set_full_tags()
        self.figure.canvas.draw()

    def set_recording(self, position):
        self.current_position = position
        self.recording_id = self.recordings_list[position]
        description = self.data_store.get_description(self.recording_id)
        self.figure.suptitle(description)
        frames, sr = self.data_store.get_recording(self.recording_id)
        self.recording_frames = frames
        self.recording_rate = sr
        self.highlights = self.data_store.get_highlights(self.recording_id)
        self.tags = self.tag_handler.get_tags(self.recording_id)
        if self.audio_stream != None:
            self.audio_stream.close()
        self.audio_stream = self.pa.open(rate=sr, format=pyaudio.paInt16, channels=1, output=True)
        nfft = int(sr / 10)
        stft = librosa.stft(frames, n_fft=nfft, hop_length=int(nfft / 2))
        npspec = np.abs(stft)[int(freqmin / 10):int(freqmax / 10)]
        self.adjusted_spectrogram = librosa.amplitude_to_db(npspec, ref=np.max)
        self.last_recording_button.set_active(position > 0)
        self.next_recording_button.set_active(position < len(self.recordings_list) - 1)
        self.full_image = self.plot_spectrogram(self.adjusted_spectrogram, 0.0, self.full_image, self.full_spectrum_ax)
        self.set_highlight(0)
        self.refresh_full_tags_highlights()


try:
    sql = ''' SELECT * FROM chris_errors
              ORDER BY recording_id, start_time  '''
    cur = db_connection.cursor()
    cur.execute(sql)
    tags = cur.fetchall()
except Exception as e:
    print(e, '\n')
    print('Failed to get highlights\n')

recording_number_of_times = {}
false_positives = {}
false_negatives = {}
for tuple in tags:
    recording_id = tuple[1]
    if recording_id in recording_number_of_times:
        recording_number_of_times[recording_id] += 1
    else:
        recording_number_of_times[recording_id] = 1
        false_negatives[recording_id] = 0
        false_positives[recording_id] = 0
    if tuple[4]:
        false_positives[recording_id] += 1
    else:
        false_negatives[recording_id] += 1
keys = list(recording_number_of_times.keys())
for id in keys:
    path = f'{segment_base_path}/{id}.m4a'
    if not os.path.exists(path):
        print(f'skipping missing recording {id} with {recording_number_of_times[id]} errors')
        recording_number_of_times.pop(id)

recordings = sorted(recording_number_of_times.keys(), key=lambda x: recording_number_of_times[x], reverse=True)
viewer = SpectrogramViewer(TagHandler(db_connection), RecordingStore(segment_base_path, false_positives, false_negatives, db_connection), recordings, 0)
