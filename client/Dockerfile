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
