import matplotlib
matplotlib.use('Qt5Agg')
import sqlite3
from tagger_base import RecordingStore, SpectrogramViewer, TagHandler

main_system = True

freqmin = 600
freqmax = 1200
bands = 60
persec = 20
window_size = 3.0
zoom_seconds = 8.0
highlight_box_color = (0.2, 0.1, 0.0, 0.1)
selection_box_color = (0.2, 0.2, 0.2, 0.2)
if main_system:
    segment_list_file = '/home/dennis/projects/timserver/february-possibles-repeated.txt'
    segment_base_path = '/mnt/tmp/morepork/february-recordings'
else:
    segment_list_file = '/home/dennis/projects/morepork/march-test-segments.txt'
    segment_base_path = '/home/dennis/projects/morepork/march-test'
db_path = '/home/dennis/projects/morepork/tim-data/audio_analysis.db'
db_connection = None
try:
    db_connection = sqlite3.connect(db_path)
except sqlite3.Error as e:
    print(e)

code_color = (0.0, 0.0, 1.0, 0.3)
code_to_hatch_and_color = { 'morepork': ('//', code_color), 'maybepork': ('xx', code_color), 'notpork': ('\\\\', code_color)}

#https://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
#https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively

try:
    sql = ''' SELECT * FROM dennis_february
              ORDER BY recording_id, slice_start_time  '''
    cur = db_connection.cursor()
    cur.execute(sql)
    tags = cur.fetchall()
except Exception as e:
    print(e, '\n')
    print('Failed to get highlights\n')
recording_highlights = {}
recording_number_of_times = {}
for tuple in tags:
    if tuple[3] > 1 and tuple[3] < 8:
        recording_id = tuple[1]
        start_time = tuple[2]
        end_time = start_time + 3
        highlight = (tuple[4], recording_id, start_time, end_time, tuple[3])
        if recording_id in recording_highlights:
            recording_highlights[recording_id].append(highlight)
            recording_number_of_times[recording_id] += tuple[3]
        else:
            recording_highlights[recording_id] = [highlight]
            recording_number_of_times[recording_id] = tuple[3]
recordings = sorted(recording_highlights.keys(), key=lambda x: recording_number_of_times[x], reverse=True)
viewer = SpectrogramViewer(TagHandler(db_connection), RecordingStore(segment_base_path, recording_highlights, db_connection), bands, freqmin, freqmax, recordings, 0)
