# Importamos del paquete
import message_filters
import rclpy
from rclpy.node import Node
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from PIL import Image

from sensor_msgs.msg import Image
import yaml

def ReadYaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        data = yaml.safe_load(stream)

    # Access specific parameters
    projection_matrix = data['projection_matrix']

    projection_matrix_data = np.array(projection_matrix['data'])

    matrix_P = []
    for i in range(projection_matrix['rows']):
        row = projection_matrix_data[i * projection_matrix['cols']:(i + 1) * projection_matrix['cols']]
        matrix_P.append(row)

    intrinsic_matrix = data['camera_matrix']

    intrinsic_matrix_data = np.array(intrinsic_matrix['data'])

    matrix_K = []
    for i in range(intrinsic_matrix['rows']):
        row = intrinsic_matrix_data[i * intrinsic_matrix['cols']:(i + 1) * intrinsic_matrix['cols']]
        matrix_K.append(row)

    distortion_coefficients = data['distortion_coefficients']
    dist_coff = []
    distortion_coefficients_data = np.array(distortion_coefficients['data'])
    for i in distortion_coefficients_data:
        dist_coff.append(i)

    return np.array(matrix_P), np.array(matrix_K), np.array(dist_coff)


def plotHomogeneousImage(H, points_original, points_other_camera, img, count, side):
    img_rgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    points_other_camera_ones = np.hstack((points_other_camera, np.ones((points_other_camera.shape[0], 1))))
    points_other_camera_transformed = np.dot(H, points_other_camera_ones.T)
    points = (points_other_camera_transformed[:2]/points_other_camera_transformed[2]).T
    for point in points:
        center_coordinates = (round(point[0]), round(point[1]))
        img_rgb = cv.circle(img_rgb, center_coordinates, 5, (0, 0, 255))  # Red color (BGR), -1 for filled circle
    for point in points_original:
        center_coordinates = (round(point[0]), round(point[1]))
        img_rgb = cv.circle(img_rgb, center_coordinates, 5, (0, 255, 0))  # Red color (BGR), -1 for filled circle
    cv.imwrite(f"images/homography_{side}_{count}.png", img_rgb)
    return


class ImgProcessing(Node):
    def __init__(self):
        super().__init__('image_rectification')
        self.left_rect = message_filters.Subscriber(self, Image, '/left/image_rect')
        self.right_rect = message_filters.Subscriber(self, Image, '/right/image_rect')
        self.left_rect  # prevent unused variable warning
        self.right_rect

        self.proj_left, self.K_left, self.dist_coeff_left = ReadYaml('data/left.yaml')
        self.proj_right, self.K_right, self.dist_coeff_right = ReadYaml('data/right.yaml')

        # Creamos el objeto ts del tipo TimeSynchronizer encargado de sincronizar los mensajes recibidos.
        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)

        # Registramos un callback para procesar los mensajes sincronizados.
        ts.registerCallback(self.callback)
        
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) 
        self.count = 0
        self.br = CvBridge() # Para convertir de mensaje de imagen de ROS a imagen de opencv
        
    
    def callback(self, left_msg, right_msg):
        ''' Matching '''
        img_left = self.br.imgmsg_to_cv2(left_msg)
        img_right = self.br.imgmsg_to_cv2(right_msg)

        # find the keypoints with ORB
        kp_left = self.orb.detect(img_left, None)
        # compute the descriptors with ORB
        kp_left, des_left = self.orb.compute(img_left, kp_left)

        # find the keypoints with ORB
        kp_right = self.orb.detect(img_right, None)
        # compute the descriptors with ORB
        kp_right, des_right = self.orb.compute(img_right, kp_right)

        # Compute matches between images
        matches = self.bf.match(des_left, des_right) 
        matches = sorted(matches, key = lambda x:x.distance)

        img_matches = cv.drawMatches(img_left,kp_left,img_right,kp_right,matches[:],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv.imwrite(f"images/matches_{self.count}.png", img_matches)

        img_matches = cv.drawMatches(img_left,kp_left,img_right,kp_right,matches[:30],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv.imwrite(f"images/30matches_{self.count}.png", img_matches)


        ''' Triangulation '''
        # Convert keypoints to numpy arrays of pixel coordinates
        points_left = np.array([kp_left[m.queryIdx].pt for m in matches], dtype=np.float32)
        points_right = np.array([kp_right[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Ensure that the points are in homogeneous coordinates
        points_left_homogeneous = np.hstack((points_left, np.ones((points_left.shape[0], 1))))
        points_right_homogeneous = np.hstack((points_right, np.ones((points_right.shape[0], 1))))

        points4d = cv.triangulatePoints(self.proj_left, self.proj_right, points_left.T, points_right.T)
        points3d = (points4d[:3, :]/points4d[3, :]).T

        #FALTA EL PLOT 3D

        '''Matriz Homográfica'''
        H, mask = cv.findHomography(points_left, points_right, cv.RANSAC,5.0)
        # La máscara es un vector de 1's y 0's que dicen qué puntos fueron filtrados y cuáles no. Hay que aplicar esa máscara a los vectores de puntos 2d y 3d
        #print(mask)
        #print(mask.shape)
        inlier_mask = mask.ravel() == 1
        points_left_inliers = points_left[inlier_mask]
        points_right_inliers = points_right[inlier_mask]
        points3d_inliers = points3d[inlier_mask]
        #print(f"Iteración {self.count}")
        
        # FALTA EL PLOT 3D
        plotHomogeneousImage(np.linalg.inv(H), points_left_inliers, points_right_inliers, img_left, self.count, "left")
        plotHomogeneousImage(H, points_right_inliers, points_left_inliers, img_right, self.count, "right")
        
        '''Mapa de Disparidad'''
        #img_left_image = Image.fromarray(img_left)
        #img_right_image = Image.fromarray(img_right)
        #print(type(img_left))
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
        disparity_map = stereo.compute(img_left, img_right)
        min_disp = disparity_map.min()
        max_disp = disparity_map.max()
        disparity_map = (disparity_map - min_disp) / (max_disp - min_disp)

        # Convert the normalized disparity map to a color image (optional)
        disparity_map_colored = cv.applyColorMap((disparity_map * 255).astype(np.uint8), cv.COLORMAP_JET)
        #cv.imwrite(f"images/disparity_map_{self.count}.png", disparity_map_colored)

        '''Reconstrucción Densa'''
        #P = K [R| t] -> P[0:3, 0:3] = K R -> R = inv (K) P[0:3,0:3]
        #             -> P[0:3,3] = K t -> t = inv(K) P[0:3,3]
        R = np.dot(np.linalg.inv(self.K_right), self.proj_right[0:3,0:3])
        t = np.dot(np.linalg.inv(self.K_right), self.proj_right[0:3,3])
        R1, R2, P1, P2, Q = cv.stereoRectify(self.K_left, self.dist_coeff_left, self.K_right, self.dist_coeff_right,img_left.shape,R,t) 
        #camera_matrix1, distCoeffs1, camera_matrix2, distCoeffs2, R, T 
        reprojection = cv.reprojectImageTo3D(disparity_map_colored, Q)
        cv.imwrite(f"images/reprojection_{self.count}.png", reprojection)

        self.count += 1
        return



def main(args=None):
    rclpy.init(args=args)

    image_rectification = ImgProcessing()

    rclpy.spin(image_rectification)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_rectification.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
