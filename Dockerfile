# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies.
# Binaries are not available for some python packages, so pip must compile them locally. This is
# why gcc, g++, and python3.8-dev are included in the list below.
# Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute capability 
# 3.5 through 9.0
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    python3.8-dev \
    python3.8-venv \
    python3.9-venv \
    wget \
    portaudio19-dev \
    libsndfile1 \
    unzip

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Download the NSF_HiFiGan model
RUN mkdir -p ~/hay_say/temp_downloads/nsf_hifigan/ && \
    wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip --directory-prefix=$HOME_DIR/hay_say/temp_downloads/nsf_hifigan/ && \
    unzip -j ~/hay_say/temp_downloads/nsf_hifigan/nsf_hifigan_20221211.zip -d ~/hay_say/temp_downloads/nsf_hifigan/ && \
    rm ~/hay_say/temp_downloads/nsf_hifigan/nsf_hifigan_20221211.zip

# Download the pre-trained Hubert model checkpoint
# Note: the wget link below consistently fails after downloading 1 GB, so I acquired the file elsewhere and am using a COPY statement instead.
# RUN wget hubert/ http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt --directory-prefix=$HOME_DIR/hay_say/so_vits_svc_4/hubert/
RUN mkdir -p ~/hay_say/temp_downloads/hubert
COPY --chown=$LIMITED_USER:$LIMITED_USER checkpoint_best_legacy_500.pt $HOME_DIR/hay_say/temp_downloads/hubert

# Create virtual environments for so-vits-svc 4.0 and Hay Say's so_vits_svc_4_server
RUN python3.8 -m venv ~/hay_say/.venvs/so_vits_svc_4; \
    python3.9 -m venv ~/hay_say/.venvs/so_vits_svc_4_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517
RUN ~/hay_say/.venvs/so_vits_svc_4/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel; \
    ~/hay_say/.venvs/so_vits_svc_4_server/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel

# Install all python dependencies for so_vits_svc_4
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# so-vits-svc 4.0 code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build
# time. See https://docs.docker.com/build/cache
RUN ~/hay_say/.venvs/so_vits_svc_4/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    ffmpeg-python==0.2.0 \
    Flask==3.0.0 \
    Flask-Cors==4.0.0 \
    gradio==3.7.0 \
    numpy==1.23.5 \
    pyworld==0.3.4 \
    scipy==1.10.0 \
    soundfile==0.12.1 \
    torch==2.1.1+cu118 \
    torchaudio==2.1.1+cu118 \
    torchcrepe==0.0.22 \
    tqdm==4.66.1 \
    scikit-maad==1.4.0 \
    praat-parselmouth==0.4.3 \
    onnx==1.15.0 \
    onnxsim==0.4.35 \
    onnxoptimizer==0.3.13 \
    fairseq==0.12.2 \
    librosa==0.9.1 \
    tensorboard==2.14.0 \
    tensorboardX==2.6.2.2 \
    transformers==4.35.2 \
    edge-tts==6.1.9 \
    PyYAML==6.0.1 \
    pynvml==11.5.0 \
    faiss-cpu==1.7.4

# Install the dependencies for the Hay Say interface code
RUN ~/hay_say/.venvs/so_vits_svc_4_server/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    hay-say-common==1.0.7 \
    jsonschema==4.19.1

# Clone so_vits_svc_4 and checkout a specific commit that is known to work with this docker
# file and with Hay Say
RUN git clone -b Moe-SVC --single-branch -q https://github.com/svc-develop-team/so-vits-svc ~/hay_say/so_vits_svc_4
WORKDIR $HOME_DIR/hay_say/so_vits_svc_4
RUN git reset --hard 153083eca42c3d17b77e20821724eced1cc49c40

# Also Clone the 4.1-Stable branch for models that require it
RUN git clone -b 4.1-Stable --single-branch -q https://github.com/svc-develop-team/so-vits-svc ~/hay_say/so_vits_svc_4_dot_1_stable
WORKDIR $HOME_DIR/hay_say/so_vits_svc_4_dot_1_stable
RUN git reset --hard 301c67b3175bf00de7bfaf26d2ab65123e3ca3c4

# Clone the Hay Say Interface code
RUN git clone -b database-cache --single-branch https://github.com/hydrusbeta/so_vits_svc_4_server ~/hay_say/so_vits_svc_4_server/

# Expose port 6576, the port that Hay Say uses for so_vits_svc_4
EXPOSE 6576

# Move the NSF_HiFiGan model to the expected directory.
RUN mv ~/hay_say/temp_downloads/nsf_hifigan/* ~/hay_say/so_vits_svc_4/pretrain/nsf_hifigan/

# Move the pre-trained Hubert model to the expected directories. The 4.1-Stable branch expects it in the /pretrain
# directory and the 4.0 branch expects it in the /hubert directory.
RUN mv ~/hay_say/temp_downloads/hubert/checkpoint_best_legacy_500.pt ~/hay_say/so_vits_svc_4/hubert/ &&\
    ln -s ~/hay_say/so_vits_svc_4/hubert/checkpoint_best_legacy_500.pt ~/hay_say/so_vits_svc_4/pretrain/

# Run the Hay Say interface on startup
CMD ["/bin/sh", "-c", "~/hay_say/.venvs/so_vits_svc_4_server/bin/python ~/hay_say/so_vits_svc_4_server/main.py --cache_implementation file"]
