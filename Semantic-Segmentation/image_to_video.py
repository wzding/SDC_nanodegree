# Import everything needed to edit/save/watch video clips
from moviepy.editor import ImageSequenceClip
def image_to_video(images_folder, file_name):
    clip = ImageSequenceClip(images_folder, load_images=True, fps=5)
    clip.write_gif(file_name,fps=5)

if __name__ == '__main__':
    images_folder = "runs/"
    file_name = "result.gif"
    image_to_video(images_folder, file_name)
