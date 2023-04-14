import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time


# 定义眼睛中点的欧几里得距离的函数
def euclidean_distance(point1, point2):
    """
    :param point1: 左边的点
    :param point2: 右边的点
    :return: 两个点的欧几里得距离
    """
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# 查找光圈位置的功能
def iris_position(iris_center, right_point, left_point):
    """
    :param iris_center: 瞳孔的中心
    :param right_point: 右眼角的点
    :param left_point: 左眼角的点
    :return:
    """
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(left_point, right_point)
    ratio = center_to_right_dist / total_distance
    iris_position = ""
    if ratio <= 0.35:
        iris_position = "left"
    elif 0.35 < ratio < 0.65:
        iris_position = "center"
    else:
        iris_position = "right"

    return iris_position, ratio


def get_position(ratio):
    """
    :param ratio: 比例
    :return: 位置
    """
    if ratio <= 0.35:
        iris_position = "left"
    elif 0.35 < ratio < 0.65:
        iris_position = "center"
    else:
        iris_position = "right"


if __name__ == '__main__':
    """
    数据初始化
    """
    mp_face_mesh = mp.solutions.face_mesh
    # variables
    frame_counter = 0
    # 左、右眼眶的关键点
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # 左、右瞳孔的关键点
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    R_H_LEFT = [33]  # 右眼的最左边
    R_H_RIGHT = [133]  # 右眼的最右边
    L_H_LEFT = [362]  # 左眼的最左边
    L_H_RIGHT = [263]  # 左眼的最右边

    # 加载视频
    cap = cv.VideoCapture('/Users/wupeng/PycharmProjects/Gaze examination/GazeTracking/Data/24_1681205543.mp4')
    """
    开始正式的凝视检查
    """
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            '''
            初始化情况：计算FPS
            '''
            # 计算FPS
            end_time = time.time() - start_time
            fps = frame_counter / end_time
            cv.putText(frame, "FPS:".format(fps), (90, 20), cv.FONT_HERSHEY_DUPLEX,
                       0.9, (147, 58, 31), 1)
            # 检测到瞳孔后开始
            if results.multi_face_landmarks:
                # print(results.multi_face_landmarks[0].landmark)
                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                     results.multi_face_landmarks[0].landmark])
                print(mesh_points.shape)
                # 1.画出瞳孔的四边形矩阵
                cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
                # 2.将正方形转换为圆形，OpenCV函数根据提供的点提供边界圆。minEnclosingCircle返回圆的中心（x，y）和半径，扭转值为浮点，需要将其转换为int。
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                # 3.将中心点转换为np数组
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                cv.putText(frame, "Left eye angle: " + str(center_left), (90, 95),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv.putText(frame, "Left eye angle: " + str(center_right), (90, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # # 4.在右眼角显示斑点
                # cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
                # cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)
                # 5.根据minEnclosingCircle的返回值绘制圆，通过circle根据圆心（x，y）和半径绘制圆的图像
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
                # 6.计算右眼比例和返回状态
                iris_pos, ratio = iris_position(center_right, mesh_points[R_H_LEFT], mesh_points[R_H_RIGHT])

                cv.putText(frame, iris_pos, (90, 60), cv.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
                print(ratio)
            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()
