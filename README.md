# RoboticaMovil-TP3

Instrucciones: 
	- Modificar en start_docker.sh la dirección del volumen para tener acceso al rosbag y a las imágenes de calibración. Éstos deben estar dentro de la carpeta volume. 
	- Buildear e ingresar al container, como dice a continuación para ejecutar los scripts del TP. 

Instrucciones para compilar y correr en docker:
	- Build: bash start_docker.sh build
	- Start: bash start_docker.sh start
	- Open container: bash start_docker.sh open

Una vez dentro del container, correr los scripts:
	- Para ejercicios 1 - 12: python3 rect_img.py
	- Para ejercicio 13: python3 trajectory.py


Los resultados se guardan como imágenes dentro de la carpeta images.
