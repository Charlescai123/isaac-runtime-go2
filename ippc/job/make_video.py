import cv2
import os
import sys
import click


@click.command()
@click.option('--img_folder', default="")
@click.option('--save_folder', default="")
@click.option('--key', default="rgb")
@click.option('--start_frame', default=int(1))
@click.option('--end_frame', default=int(100))
def run(**kwargs):
    vis_folder = kwargs["img_folder"]
    start_frame = kwargs["start_frame"]
    end_frame = kwargs["end_frame"]
    save_folder = kwargs["save_folder"]
    key = kwargs["key"]
    os.makedirs(save_folder, exist_ok=True)

    vtype = "video_" + key
    images_dir = os.path.join(vis_folder)
    frame = cv2.imread(os.path.join(images_dir, f'{key}_0.png'))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = os.path.join(save_folder, vtype + '.mp4')
    video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))
    for number in range(start_frame, end_frame + 1):
        imname = f'{key}_{number}.png'
        video.write(cv2.imread(os.path.join(images_dir, imname)))

    cv2.destroyAllWindows()
    video.release()
    print('video_saved.')


if __name__ == '__main__':
    run()
