# How to run
docker run -it --network=host --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/pytorch:latest

# Inside the docker
git clone https://github.com/pssiva152/debug
cd debug
pip3 install sentence_transformers
pip3 install pandas
pip3 uninstall numpy # While installing pandas it will bump up the numpy version so it will throw incompatible errors, so we will uninstall numpy-2.2.3 and install 1.22.4
pip3 install numpy==1.22.4
python3 bert.py
python3 roberta.py
