import os


def convert_avi_to_mp4(avi_file_path):
    os.popen(
        "ffmpeg -i '" + avi_file_path + "' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '" +
        avi_file_path.split(".")[0] + ".mp4'")


if __name__ == '__main__':

    video_folder = os.getcwd() + "/data"
    videos = os.listdir(video_folder)

    avoid_files = [".DS_Store"]
    for video in videos:
        if (video not in avoid_files):
            print(video)
            convert_avi_to_mp4("data/" + video)
