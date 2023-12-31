# Importamos del paquete
import message_filters
import rclpy
from rclpy.node import Node
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv_bridge import CvBridge
from PIL import Image
from camera_pose_visualizer import CameraPoseVisualizer
from sensor_msgs.msg import Image
import yaml
import open3d as o3d



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
    dist_coeff = np.empty((1, 5), dtype=np.float64)
    distortion_coefficients_data = np.array(distortion_coefficients['data'])
    for i, coeff in enumerate(distortion_coefficients_data):
        dist_coeff[0,i] = coeff

    return np.array(matrix_P), np.array(matrix_K), dist_coeff


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
    cv.imwrite(f"images/homography/{side}_{count}.png", img_rgb)
    return

def plotKeypointsImage(keypoints, img, side, count):
    img_rgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    for point in keypoints:
        center_coordinates = (round(point.pt[0]), round(point.pt[1]))
        img_rgb = cv.circle(img_rgb, center_coordinates, 5, (0, 0, 255))  # Red color (BGR), -1 for filled circle
    cv.imwrite(f"images/keypoints/{count}_{side}.png", img_rgb)
    return

def plotCameraPoses(R1, t1, R2, t2, count):

    # # Aplicamos la transformación relativa para obtener la pose de la segunda cámara
    # R2 = np.dot(R2, R1)  # Combinamos las rotaciones
    # t2 = t2 + np.dot(R1, t2)  # Combinamos las traslaciones
    
    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [-5, 5])
    P1 = np.vstack((np.hstack((R1,t1)), np.array([0,0,0,1])))
    visualizer.extrinsic2pyramid(P1, 'c', 1)
    P2 = np.vstack((np.hstack((R2,t2)), np.array([0,0,0,1])))
    visualizer.extrinsic2pyramid(P2, 'r', 1)
    # Guardar plot
    plt.savefig(f"images/poses_dos_camaras/{count}.png")
    plt.close()
    return

def plotCameraTrajectory(R_list, t_list, count):

    # # Aplicamos la transformación relativa para obtener la pose de la segunda cámara
    # R2 = np.dot(R2, R1)  # Combinamos las rotaciones
    # t2 = t2 + np.dot(R1, t2)  # Combinamos las traslaciones
    
    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [-5, 5])
    for ind, _ in enumerate(R_list):
        P = np.vstack((np.hstack((R_list[ind],t_list[ind])), np.array([0,0,0,1])))
        visualizer.extrinsic2pyramid(P, 'c', 1)
    # Guardar plot
    plt.savefig(f"images/trajectory/trayectoria_{count}.png")
    plt.close()
    return

def savePointCloud(Points3D, folder, count):
    pcd = o3d.geometry.PointCloud()

    vertices = o3d.utility.Vector3dVector(Points3D)
    pcd.points = vertices
    o3d.io.write_point_cloud(f"images/{folder}/{count}.ply", pcd)
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
        self.distance_between_cameras = abs(self.proj_right[0,3])
        self.focal_length_of_camera =(self.K_left[0,0]+self.K_left[1,1])/2
        
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
        keypoints_left = self.orb.detect(img_left, None)
        # compute the descriptors with ORB
        keypoints_left, des_left = self.orb.compute(img_left, keypoints_left)

        # find the keypoints with ORB
        keypoints_right = self.orb.detect(img_right, None)
        # compute the descriptors with ORB
        keypoints_right, des_right = self.orb.compute(img_right, keypoints_right)

        plotKeypointsImage(keypoints_left, img_left, "left", self.count)
        plotKeypointsImage(keypoints_right, img_right, "right", self.count)

        # Compute matches between images
        matches = self.bf.match(des_left, des_right) 
        matches = sorted(matches, key = lambda x:x.distance)

        img_matches = cv.drawMatches(img_left,keypoints_left,img_right,keypoints_right,matches[:],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(f"images/matches/{self.count}.png", img_matches)

        img_matches = cv.drawMatches(img_left,keypoints_left,img_right,keypoints_right,matches[:30],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(f"images/30matches/{self.count}.png", img_matches)
        

        ''' Triangulation '''
        # Convert keypoints to numpy arrays of pixel coordinates
        points_left = np.array([keypoints_left[m.queryIdx].pt for m in matches], dtype=np.float32)
        points_right = np.array([keypoints_right[m.trainIdx].pt for m in matches], dtype=np.float32)

        points4d = cv.triangulatePoints(self.proj_left, self.proj_right, points_left.T, points_right.T)
        points3d = (points4d[:3, :]/points4d[3, :]).T

        savePointCloud(points3d, "points3dstereo", self.count)

        '''Matriz Homográfica'''
        H, mask = cv.findHomography(points_left, points_right, cv.RANSAC,2.0)
        # La máscara es un vector de 1's y 0's que dicen qué puntos fueron filtrados y cuáles no. Hay que aplicar esa máscara a los vectores de puntos 2d y 3d
        #print(mask)
        #print(mask.shape)
        inlier_mask = mask.ravel() == 1
        points_left_inliers = points_left[inlier_mask]
        points_right_inliers = points_right[inlier_mask]
        points3d_inliers = points3d[inlier_mask]
        #print(f"Iteración {self.count}")
        
        savePointCloud(points3d_inliers, "points3dstereo_inliers", self.count)

        plotHomogeneousImage(np.linalg.inv(H), points_left_inliers, points_right_inliers, img_left, self.count, "left")
        plotHomogeneousImage(H, points_right_inliers, points_left_inliers, img_right, self.count, "right")
        
        '''Mapa de Disparidad'''
        #img_left_image = Image.fromarray(img_left)
        #img_right_image = Image.fromarray(img_right)
        #print(type(img_left))
        stereo = cv.StereoBM.create(numDisparities=0, blockSize=15)
        disparity_map = stereo.compute(img_left, img_right)
        min_disp = disparity_map.min()
        max_disp = disparity_map.max()
        disparity_map = ((disparity_map - min_disp) / (max_disp - min_disp)).astype(np.float32)

        # Convert the normalized disparity map to a color image (optional)
        disparity_map_colored = cv.applyColorMap((disparity_map * 255).astype(np.uint8), cv.COLORMAP_JET)
        cv.imwrite(f"images/disparity_map/{self.count}.png", disparity_map_colored)

        '''Reconstrucción Densa'''
        #P = K [R| t] -> P[0:3, 0:3] = K R -> R = inv (K) P[0:3,0:3]
        #             -> P[0:3,3] = K t -> t = inv(K) P[0:3,3]
        R = np.dot(np.linalg.inv(self.K_right), self.proj_right[0:3,0:3])
        t = np.dot(np.linalg.inv(self.K_right), self.proj_right[0:3,3])

        R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(self.K_left, self.dist_coeff_left, self.K_right, self.dist_coeff_right,img_left.shape,R,t) 
        #camera_matrix1, distCoeffs1, camera_matrix2, distCoeffs2, img, R, T 
        reprojection = cv.reprojectImageTo3D(disparity_map, Q)
        reprojectionPoints = reprojection.reshape(-1, 3)

        Points3D_filtered = []
        for point in reprojectionPoints:
            if abs(point[0])==np.inf or abs(point[1])==np.inf or abs(point[2])==np.inf: 
                pass
            else:
                Points3D_filtered.append(point)
                
        cv.imwrite(f"images/reprojection/{self.count}.png", reprojection)
        
        savePointCloud(Points3D_filtered, "points3dstereo_reprojection", self.count)


        '''Estimación de pose'''
        # Como las matrices de las camaras no son exactamente iguales, como asume findEssentialMat, necesitamos compensar la distorsión.
        # En nuestro caso las imágenes ya están rectificadas, por lo que esta compensación ya está hecha???
        E, _ = cv.findEssentialMat(points1=points_left_inliers, points2=points_right_inliers, cameraMatrix1=self.K_left, distCoeffs1=self.dist_coeff_left, 
                       cameraMatrix2=self.K_right, distCoeffs2=self.dist_coeff_right)
        # E, R y t se le dan a cv.RecoverPose como estimaciones iniciales, y después la función devuelve los valores finales estimados.
        _, E, R, t, _ = cv.recoverPose(points1=points_left_inliers, points2=points_right_inliers, cameraMatrix1=self.K_left, distCoeffs1=self.dist_coeff_left, 
                                        cameraMatrix2=self.K_right, distCoeffs2=self.dist_coeff_right, E=E, R=R, t=t, method=cv.RANSAC)

        plotCameraPoses(np.eye(3), np.zeros((3,1)), R,  t.reshape((3, 1)), self.count)
        
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
