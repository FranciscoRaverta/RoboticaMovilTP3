FROM osrf/ros:humble-desktop-full

WORKDIR /usr/src/

RUN sudo apt update && sudo apt install python3-pip -y && pip install open3d