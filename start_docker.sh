#!/bin/bash
set -eo pipefail
DEFAULT_HOST="$WO_HOST"
DEFAULT_DIR="$WO_DIR"

# Parse args for overrides
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --hostname)
    export WO_HOST="$2"
    shift # past argument
    shift # past value
    ;;
	--dir)

    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameter

usage(){
  echo "Usage: $0 <command>"
  echo
  echo "This program helps to manage the setup/teardown of the docker containers for running Piwigo. We recommend that you read the full documentation of docker at https://docs.docker.com if you want to customize your setup."
  echo 
  echo "Command list:"
  echo "	start [options]		Start PointLearning"
  echo "	stop			Stop PointLearning"
  echo "	down			Stop and remove PointLearning's docker containers"
  echo "	rebuild			Rebuild docker image and perform cleanups"
  echo "        open			Open terminal to run scripts and whatever"
  echo ""
  #echo "Options:"
  #echo "	--hostname	<hostname>	Set the hostname that PointLearning will be accessible from (default: $DEFAULT_HOST)"
  #echo "	---dir	<path>	Path where data will be persisted (default: $DEFAULT_DIR (docker named volume))"
  exit
}


run(){
  echo "$1"
  eval "$1"
}

start(){
	run "docker run --rm --name robomoviltp3 -v ./volume:/usr/src/ros2_ws/src -it -d robomoviltp3"
  # -it is for having an interactive terminal, and -d is enable reenter after closing the terminal
}

down(){
	# Completar
  run "docker rm robomoviltp3"
}

rebuild(){
  down
  run "docker build -t robomoviltp3 . "
}

build(){
  run "docker build --no-cache -t robomoviltp3 ."
}

open(){
  run "docker exec -it robomoviltp3 bash"
}


if [[ $1 = "start" ]]; then
	start
elif [[ $1 = "stop" ]]; then
	echo "Stopping robomoviltp3..."
	run "docker stop robomoviltp3"
elif [[ $1 = "down" ]]; then
	echo "Tearing down robomoviltp3..."
	down
elif [[ $1 = "rebuild" ]]; then
	echo  "Rebuilding robomoviltp3..."
	rebuild
elif [[ $1 = "build" ]]; then
        echo  "Building robomoviltp3..."
	build
elif [[ $1 = "open" ]]; then
	echo "Opening terminal..."
	open
else
	usage
fi


