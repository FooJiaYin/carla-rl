# CARLA Setup

Follow the steps in [Running in a Docker - CARLA Simulator](https://carla.readthedocs.io/en/latest/build_docker/).
1. [Install docker](#Install-docker)
2. [Install nvidia-docker](#Install-nvidia-docker)
3. [Running CARLA RL](#Running-CARLA-RL)

:::info
Add `sudo` before any `apt` or `docker` comment, or if permission denied message received.
:::

## Install docker
Refer to: https://docs.docker.com/engine/install/ubuntu/
```
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```
If there are any package not updgraded, run:
```
$ sudo autoremove
$ sudo apt upgrade
```
Next, 
```
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Run a test
```
$ sudo docker run hello-world
```

## Install nvidia-docker
Refer to: https://github.com/NVIDIA/nvidia-docker
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
Test nvidia-smi with the latest official CUDA image
```
$ sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```
:::info
In new version, use `docker run` instead of `nvidia-docker run`.
Remember to add `sudo`.
:::

## Running CARLA RL
Refer to: 
https://carla.readthedocs.io/en/latest/build_docker/
https://github.com/carla-rl-gym/carla-rl
```
$ sudo docker pull carlasim/carla:0.8.2
```
Clone the carla-rl github repo
```
$ git clone https://github.com/foojiayin/carla-rl.git
$ cd carla-rl
```
:::info
My fork: https://github.com/foojiayin/carla-rl.git (Added some unstated dependencies)
Original repo: https://github.com/carla-rl-gym/carla-rl.git (If you clone this repo, you might need to **[modify some files](#Modified-files)**
:::
### Building server 
Build and run server in docker.
```
$ sudo docker build server -t carla-server
$ sudo docker run --rm -it --gpus all -p 2000-2002:2000-2002 carlasim/carla:0.8.2 /bin/bash
```

### Running server
Now we are in a container with username carla.
```
$ ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -ResX=800 -ResY=600 -carla-world-port=2000
```
- `Town01` can be replaced with other map
- port can be changed, too

#### Debug
- 
    :::danger
    xdg-user-dir: not found
    :::
    ***Solution***
    Run carla version 0.8.2 instead of 0.9.9 (latest)
    ```
    sudo docker run --rm -it --gpus all -p 2000-2002:2000-2002 carlasim/carla:0.8.2 /bin/bash
    ```
- 
    :::danger
    docker: Error response from daemon: OCI runtime create failed
    unsatisfied condition: cuda>=10.2, please update your driver to a newer version, or use an earlier cuda container
    :::
    ***Solution***
    Upgrade CUDA version： https://developer.nvidia.com/cuda-downloads
    Choose OS version (BigWhite: Linux > x86-64 > Ubuntu > 16.04)
- 
    :::danger
    segmentation fault SIGSEGV: invalid attempt to read memory at address 0x0000000000000010 
    :::
    ***Solution***
    When [build and run the server](#Building-server): add `--gpus all`
    ```
    $ ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -ResX=800 -ResY=600 -carla-world-port=2000
    ```    
    
### Build and run client
- Before building client, please apply **[the following changes](#Modified-files)** if you clone the repo from https://github.com/carla-rl-gym/carla-rl
- Next, build and run the client.
    ```
    $ sudo docker build client -t carla-client
    $ sudo docker run -it --network=host --gpus all -v $PWD:/app carla-client /bin/bash
    ```
#### Debug
- During pygame installation
    :::danger
    Unable to run "sdl-config". Please make sure a development version of SDL is installed. 
    :::
    ***Solution*** 
    In `DockerFile`, add
    ```
    RUN sudo apt-get install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev 
    RUN sudo apt-get install -y libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev
    RUN sudo apt-get install -y python3-dev python3-numpy
    ```
    
### Training in the client
Now we are in a container with username `root`.
- Make sure the server is on. Next, run the client.
    ```
    $ python client/train.py --config client/config/base.yaml
    ```
    :::info
    If the server port is not `2000`, please add `--starting-port=`, follow by the port number
    :::
    
#### Debug
- 
    :::danger
    TCP connection closed *(xN)*
    :::
    ***Solution***
    Please check that the server is running, and the `--starting-port` argument in client is consistent with the `--carla-world-port` in server.
    
- If error message (after client is successfully connected)
    :::danger
    assert dtype is not None, 'dtype must be explicitly provided. '
    :::
    ***Solution***
    Install gym version 0.10.8 instead of gym: 
    In `carla-rl/client/requirements.txt`: replace `gym` with `gym==0.10.8`
    
- 
    :::danger
    no module named 'cv2'
    :::
    ***Solution***
    Install opencv.
    In `carla-rl/client/requirements.txt`: add `opencv-python`
- 
    :::danger
    from .cv2 import * ImportError: libSM.so.6: cannot open shared object file: No such file or directory
    :::
    ***Solution***
    In `carla-rl/client/DockerFile`: add 
    ```
    RUN sudo apt-get update
    RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev
    ```
## Modified files

- carla-rl/client/DockerFile
```
FROM anibali/pytorch

USER root
# Install vim for local development
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "ffmpeg"]

# Add requirement.txt first for caching purposes.
COPY requirements.txt /app
RUN sudo apt-get install -y mercurial 
RUN hg clone https://bitbucket.org/pygame/pygame
RUN cd pygame
RUN sudo apt-get install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev 
RUN sudo apt-get install -y libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev
RUN sudo apt-get install -y python3-dev python3-numpy
RUN sudo apt-get update
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install -r requirements.txt

# Running a terminal lets you run any script.
CMD /bin/bash
```
- carla-rl/client/requirements.txt
```
Pillow
numpy
protobuf
pygame
matplotlib
future
tensorboardX
scikit-video
pyyaml
gym==0.10.8
IPython
opencv-python
```
- carla-rl/client/carla/client.py
```python=169
    def _read_sensor_data(self):
        while True:
            data = self._stream_client.read()
            if not data:
                return
                # raise StopIteration
            yield self._parse_sensor_data(data)
```


#### Debug
:::danger
docker: Error response from daemon: Unknown runtime specified nvidia
:::




## Useful Commands in Docker
- Get container id or check status
    ```
    sudo docker ps       # Check current running containers
    sudo docker ps -a    # Check all containers
    ```
    ![](https://i.imgur.com/oMkSIYr.png)

- Copy files into container (do this outside container)
    ```
    sudo docker cp ${file-to-copy} ${container-id}:\${dest-location-in-docker}
    
    ## Example
    # Copy file from computer to container
    sudo docker cp copyme.txt c4d9f55f4db4:/home/carla/PythonClient

    # Copy file from container to computer
    sudo docker cp c4d9f55f4db4:/home/carla/CarlaUE4.sh  ~/carla-rl/
    ```
    - Do no use '~/' for home directory in container, use full directory, ex: ''/home/carla' (for server)
    - The files in client '/app' refers to the computer '/carla-rl', do not need to use `docker cp`
    
- Open terminal in a running container instance
    ```
    sudo docker exec -it ${container-id} bash
    
    ## Example
    sudo docker exec -it c4d9f55f4db4 bash
    ```


## Output results
debug videos under `carla-rl/outputs/video` (show the last few minutes before collision / reach goal)

## Reproduce Benchmark

### A2C
To reproduce results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/a2c.yaml`

### ACKTR
To reproduce results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/acktr.yaml`

### PPO
To reproduce results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/ppo.yaml`

### On-Policy HER
To reproduce results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/her.yaml`



