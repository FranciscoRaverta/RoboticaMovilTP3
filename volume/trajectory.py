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

    limits = 15
    
    visualizer = CameraPoseVisualizer([-limits, limits], [-limits, limits], [-limits, limits])

    vertices = []
    faces = []
    ind = 0
    for ind, _ in enumerate(R_list):
        P = np.vstack((np.hstack((R_list[ind],t_list[ind])), np.array([0,0,0,1])))
        visualizer.extrinsic2pyramid(P, 'c', 0.3)

        vertices.extend([array.tolist() for array in visualizer.verts])
        faces.extend([[f + 5 * ind for f in sublist] for sublist in visualizer.faces] )

    # Guardar plot
    plt.savefig(f"images/trajectory/trayectoria_{count}.png")
    
    # Create a Trimesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Save the mesh as an .obj file
    o3d.io.write_triangle_mesh(f"images/trajectory/trayectoria_{count}.obj", mesh)

    plt.close()
    return

def savePointCloud(Points3D, count):
    pcd = o3d.geometry.PointCloud()

    vertices = o3d.utility.Vector3dVector(Points3D)
    pcd.points = vertices
    o3d.io.write_point_cloud(f"images/trajectory3DPoints/{count}.ply", pcd)
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
        self.bf12 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.bf23 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self.count = 0
        self.br = CvBridge() # Para convertir de mensaje de imagen de ROS a imagen de opencv
        
        self.t_left = []
        self.R_left = []
         
        self.img_left_prev = None
        self.img_right_prev = None
        self.matches_prev = None
        self.keypoints_left_prev = None
        self.keypoints_right_prev = None
        
        
    
    def callback(self, left_msg, right_msg):
        ''' Matching '''
        img_left = self.br.imgmsg_to_cv2(left_msg)
        img_right = self.br.imgmsg_to_cv2(right_msg)
        cv.imwrite(f"images/left_img/{self.count}.png", img_left)

        # find the keypoints with ORB
        keypoints_left = self.orb.detect(img_left, None)
        # compute the descriptors with ORB
        keypoints_left, des_left = self.orb.compute(img_left, keypoints_left)

        # find the keypoints with ORB
        keypoints_right = self.orb.detect(img_right, None)
        # compute the descriptors with ORB
        keypoints_right, des_right = self.orb.compute(img_right, keypoints_right)

        # Compute matches between images
        matches = self.bf.match(des_left, des_right) 
        matches = sorted(matches, key = lambda x:x.distance)

        '''Estimación trayectoria cámara left'''
        if self.count == 0:
            t_left = np.zeros((3,1))
            R_left = np.eye(3)
        else:
            # find the keypoints with ORB
            keypoints_left_prev = self.orb.detect(self.img_left_prev, None)
            # compute the descriptors with ORB
            keypoints_left_prev, des_left_prev = self.orb.compute(self.img_left_prev, keypoints_left_prev)
            # find the keypoints with ORB
            keypoints_right_prev = self.orb.detect(self.img_right_prev, None)
            # compute the descriptors with ORB
            keypoints_right_prev, des_right_prev = self.orb.compute(self.img_right_prev, keypoints_right_prev)


            ###############################################################################################################
            '''Matcheo de a 3 imágenes (Imagen1 = left actual, Imagen2 = left anterior, Imagen3 = right anterior)'''
            matches12 = self.bf12.match(des_left, des_left_prev)
            matches23 = self.matches_prev

            # Matches comunes a las 3 imágenes
            common_matches12 = []
            common_matches23 = []
            for match12 in matches12:
                for match23 in matches23:
                    if match23.queryIdx == match12.trainIdx:
                        common_matches12.append(match12)
                        common_matches23.append(match23)

            img_matches12 = cv.drawMatches(img_left,keypoints_left,self.img_left_prev,keypoints_left_prev,common_matches12[:],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(f"images/matches12/{self.count}.png", img_matches12)
            img_matches23 = cv.drawMatches(self.img_left_prev,keypoints_left_prev,self.img_right_prev,keypoints_right_prev,common_matches23[:],outImg=None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(f"images/matches23/{self.count}.png", img_matches23)
            ###############################################################################################################
            ''' Triangulation '''
            # Convert keypoints to numpy arrays of pixel coordinates
            points_left = np.array([keypoints_left[m.queryIdx].pt for m in common_matches12], dtype=np.float32)
            points_left_prev = np.array([self.keypoints_left_prev[m.queryIdx].pt for m in common_matches23], dtype=np.float32)
            points_right_prev = np.array([self.keypoints_right_prev[m.trainIdx].pt for m in common_matches23], dtype=np.float32)

            points_4d = cv.triangulatePoints(self.proj_left, self.proj_right, points_left_prev.T, points_right_prev.T)
            points_3d = (points_4d[:3, :]/points_4d[3, :]).T


            H12, mask = cv.findHomography(points_left, points_left_prev, cv.RANSAC,3.0)
            # La máscara es un vector de 1's y 0's que dicen qué puntos fueron filtrados y cuáles no. Hay que aplicar esa máscara a los vectores de puntos 2d y 3d
            # Con H12 filtramos outliers en 1 y 2, pero también en 3, porque los matcheos son de a 3.
            inlier_mask = mask.ravel() == 1
            tol = 1e10
            for ind, flag in enumerate(inlier_mask):
                if flag:
                    point_3d = points_3d[ind]
                    if point_3d[0] > tol or point_3d[1] > tol or point_3d[2] > tol:
                        inlier_mask[ind] = False
            points_left_inliers = points_left[inlier_mask]
            points_3d_inliers = points_3d[inlier_mask]
            #print("Points 3D:  \n ",points_3d_inliers)
            #print("Points Left:  \n ", points_left_inliers)
    
            savePointCloud(points_3d_inliers, self.count)
            retval, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=points_3d_inliers, imagePoints=points_left_inliers, cameraMatrix=self.K_left, distCoeffs=self.dist_coeff_left)
            R, _ = cv.Rodrigues(rvec)
            
            # Hasta acá el cálculo me devuelve R = R21, tvec = t21, es decir, R,tvec son de la cámara left relativo a la cámara left anterior, no globales.
            # Aplicamos la transformación relativa para obtener la pose de la segunda cámara
            # R_left = np.dot(R12, R_left_prev)  # Combinamos las rotaciones
            # t_left = t_left_prev + np.dot(R_left_prev, t12)  # Combinamos las traslaciones

            # R_left = np.dot(R, self.R_left[-1])
            R_left = np.dot(self.R_left[-1], R)
            t_left = self.t_left[-1] + np.dot(self.R_left[-1], tvec.reshape((3, 1)))
        


        self.t_left.append(t_left)
        self.R_left.append(R_left)
        plotCameraTrajectory(self.R_left, self.t_left, self.count)

        self.img_left_prev = img_left
        self.img_right_prev = img_right
        self.matches_prev = matches
        self.keypoints_left_prev = keypoints_left
        self.keypoints_right_prev = keypoints_right
        
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
