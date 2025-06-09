from tomi import TOMISong, SongGenre, KeyMode


tomi_song = TOMISong(
    song_name="tomi_song",
    song_genre=SongGenre.Pop,
    song_blocks=None, # You can use the path of a previously generated JSON file here.
    key=KeyMode('C', 'major')
)


tomi_song.gen(stream_output=False, open_editor=False)
