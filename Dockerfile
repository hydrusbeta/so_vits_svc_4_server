# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies.
# Binaries are not available for some python packages, so pip must compile them locally. This is
# why gcc, g++, and python3.8-dev are included in the list below.
# Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute capability 
# 3.5 through 9.0
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone.
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    python3.8-dev \
    python3.8-venv \
    python3.9-venv \
    wget \
    portaudio19-dev \
    libsndfile1

# todo: Is there a better way to refer to the home directory (~)?
ARG HOME_DIR=/root

# download so_vits_svc_4 and checkout a specific commit that is known to work with this docker 
# file and with Hay Say
RUN git clone -b 4.0 --single-branch -q https://github.com/svc-develop-team/so-vits-svc ~/hay_say/so_vits_svc_4
WORKDIR $HOME_DIR/hay_say/so_vits_svc_4
RUN git reset --hard 489937389cb3c4b123de47582e7d1060497870f9

# Also download the 4.0-Vec768-Layer12 branch for models that require it
RUN git clone -b 4.0-Vec768-Layer12 --single-branch -q https://github.com/svc-develop-team/so-vits-svc ~/hay_say/so_vits_svc_4_Vec768-Layer12
WORKDIR $HOME_DIR/hay_say/so_vits_svc_4_Vec768-Layer12
RUN git reset --hard 52c5ea8c46a068794db1001ca08acde3711d7c90

# Create virtual environments for so-vits-svc 4.0 and Hay Say's so_vits_svc_4_server
RUN python3.8 -m venv ~/hay_say/.venvs/so_vits_svc_4; \
    python3.9 -m venv ~/hay_say/.venvs/so_vits_svc_4_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while 
# we're at it to handle modules that use PEP 517
RUN ~/hay_say/.venvs/so_vits_svc_4/bin/pip install --no-cache-dir --upgrade pip wheel; \
    ~/hay_say/.venvs/so_vits_svc_4_server/bin/pip install --no-cache-dir --upgrade pip wheel

# Install all python dependencies for so_vits_svc_4
RUN ~/hay_say/.venvs/so_vits_svc_4/bin/pip install --no-cache-dir -r ~/hay_say/so_vits_svc_4/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# Download the pre-trained Hubert model checkpoint
# Note: the wget link below consistenly fails after downloading 1 GB, so I acquired the file elsewhere and am using a COPY statement instead.
# RUN wget hubert/ http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt --directory-prefix=/root/hay_say/so_vits_svc_4/hubert/
COPY checkpoint_best_legacy_500.pt $HOME_DIR/hay_say/so_vits_svc_4/hubert/

# Download the Hay Say Interface code and install its dependencies
RUN git clone https://github.com/hydrusbeta/so_vits_svc_4_server ~/hay_say/so_vits_svc_4_server/ && \
	~/hay_say/.venvs/so_vits_svc_4_server/bin/pip install --no-cache-dir -r ~/hay_say/so_vits_svc_4_server/requirements.txt

# Expose port 6576, the port that Hay Say uses for so_vits_svc_4
EXPOSE 6576

# Run the Hay Say interface on startup
CMD ["/bin/sh", "-c", "/root/hay_say/.venvs/so_vits_svc_4_server/bin/python /root/hay_say/so_vits_svc_4_server/main.py"]
