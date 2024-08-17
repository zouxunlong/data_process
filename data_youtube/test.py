from pytube import YouTube
import os

try:
    directory="./"
    filename="5GfKhwJ8FiI.wav"
    yt = YouTube("https://www.youtube.com/watch?v=5GfKhwJ8FiI")
    print("youtubeed", flush=True)
    stream = yt.streams.filter(only_audio=True).first()
    print("streammed", flush=True)
    stream.download(directory, filename=filename)
    print("downloaded", flush=True)
except Exception as e:
    if "Forbidden" in str(e):
        os.remove(os.path.join(directory, filename))
        print("{} ERROR: {}".format(filename, e), flush=True)
    else:
        print("{} ERROR: {}".format(filename, e), flush=True)
