from pytube import YouTube
import os



def down(id):
    try:
        directory="./"
        filename=id+".wav"
        yt = YouTube("https://www.youtube.com/watch?v="+id)
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


if __name__ == "__main__":
    import fire
    fire.Fire(down)