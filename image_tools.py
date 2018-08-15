
import cv2
import os


class ImageTools(object):
    """
    Image tools class, it concentrates frames (image files) to make a video
    """

    def __init__(self, default_fps=18):
        self._desired_fps = default_fps


    @staticmethod
    def frames_compare(x, y):
        """
        compare callback function

        :param x: rosbag name 1
        :param y: rosbag name 2
        :return:
        """
        digits_x = int(x.split('.')[0])
        digits_y = int(y.split('.')[0])

        if digits_x < digits_y:
            return -1
        elif digits_x > digits_y:
            return 1
        else:
            return 0

    @staticmethod
    def video2frames(video_full_path, dump_frames_root_dir):
        """
        parse the video and dump it into frames

        :param video_full_path:
        :param dump_frames_root_dir:
        :return:
        """
        if not os.path.exists(dump_frames_root_dir):
            os.mkdir(dump_frames_root_dir)

        cap = cv2.VideoCapture(video_full_path)

        index = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            index += 1
            print index, frame.shape

            frame_name = '%d.jpg' % index
            output_img_file = os.path.join(dump_frames_root_dir, frame_name)
            cv2.imwrite(output_img_file, frame)

        cap.release()
        print 'Total number of frames = {}'.format(index)
        return

    def frames2video(self, load_frames_root_dir, video_reconstruct_full_path):
        """
        reconstruct the video from different frames

        :param load_frames_root_dir:
        :param video_reconstruct_full_path:
        :return:
        """
        im_names = os.listdir(load_frames_root_dir)
        frames_num = len(im_names)
        im_names.sort(self.frames_compare)

        temp = cv2.imread(os.path.join(load_frames_root_dir, '1.jpg'), cv2.IMREAD_COLOR)
        size = (temp.shape[1], temp.shape[0])
        print 'frame size = {} x {}'.format(size[0], size[1])
        fourcc = cv2.cv.FOURCC(*'XVID') #cv2.VideoWriter_fourcc(*'XVID')
        videoWriter = cv2.VideoWriter(video_reconstruct_full_path, fourcc, self._desired_fps, size)

        index = 0
        for im_name in im_names:
            img_file = os.path.join(load_frames_root_dir, im_name)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            videoWriter.write(img)
            index += 1
            print 'Processing frame {}/{}'.format(index, frames_num)

        videoWriter.release()
        print "released"
        return


if __name__ == '__main__':
    """
        Demo of video extraction from ROSBags
    """
    imageTools = ImageTools(default_fps=8)

    """
        Demo of frames to video
    """

    imageTools.frames2video(
        load_frames_root_dir="frames/",
        video_reconstruct_full_path='./trajectory_prediction.avi'
    )
