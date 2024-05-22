

# MLOPS conda Environment Preparation Cloud or Local

**Note**: 
- You don't have to rent an instance in the cloud. Skip the **`Cloud Steps`** go to **`Environment Steps`**, then You can follow the same instructions 
for setting up your local environment. 
- `Recommended development environment: Mac, Linux, WSL on Windows`. Another Option: Windows.



## Cloud Steps:


### Cloud Step 1: Create Cloud Account for Instance

- #### Create AWS Account for EC2 Instance.
    - https://portal.aws.amazon.com/billing/signup#/start/email
    - https://console.aws.amazon.com/console/home
    - Build EC2 Instance: Ubuntu Linux with free tier configuration
        - Give Name EC2 Instance: mlops-... 
        - t2.micro (free tier)
        - Build or Select SSH Key: *.pem file 
        - Disk Space: 30 GB (max free tier)
    - Start Created EC2 Instance

- #### Create an Deploifai (on AWS, Azure, or GCP) Account.
    - https://github.com/98sean98/mlops-zoomcamp/blob/main/01-intro/deploifai-server/readme.md

- #### Create an Google Cloud Account.
    - https://cloud.google.com/
    - https://github.com/piyush-an/MLOps-ZoomCamp/blob/main/01-Introduction/infrastructure.md


### Cloud Step 2: Connect Amazon Ubuntu EC2 Instance

#### Cloud Step 2.1: Open Local Terminal (or GIT Bash)

#### Cloud Step 2.2: (İf Required) Update existing local packages 

```sh
# Update existing local packages
sudo apt update -y 
```
```sh
# Upgrade Linux
sudo apt upgrade -y
```

#### Cloud Step 2.3: Check local home folder

```sh
# Go to local HOME folder
cd
```
```sh  
# Check local directory is in HOME folder
pwd
```
```sh  
# Check local HOME folders inside
ls -al
```

#### Cloud Step 2.4: Create local ".ssh" folder, then copy in ssh key "*.pem" file

```sh
# Create local .ssh folder in HOME folder
mkdir -p ~/.ssh
```
```sh  
# Go to local .ssh folder
cd ~/.ssh/
```
```sh  
# Copy local *.pem key to in .ssh folder
cp "pem key path" .
```
```sh 
# or Copy Manual local mac
open .
```
```sh 
# or Copy Manual local wsl
explorer.exe .
```
```sh 
# or Copy Manual local linux
sudo apt install nautilus -y
nautilus .
```

**Note**: If you get `It is required that your private key files are NOT accessible by others. This private key will be ignored.` error, you should change permits on the downloaded file to protect your private key:

```sh 
# for local ssh pem key permission error
chmod 400 ~/.ssh/name-of-your-private-key-file.pem
```

- (İf Required) Permission to a specific USER
```sh 
# giving permission to a specific user
sudo chown -R username: ~/.ssh/name-of-your-private-key-file.pem
```

#### Cloud Step 2.5: Configure local ssh Connection

```sh
# Build local ssh config File with nano editor in .ssh folder
nano ~/.ssh/config
```
```sh 
# Copy Configuration in local nano editor, then Save it!
# ssh connection calling name
Host mlops-zoomcamp
    # username AWS EC2
    User ubuntu
    # Public IP, it changes when Source EC2 is turned off.
    HostName <instance-public-IPv4-addr>
    # Private SSH key file path
    IdentityFile ~/.ssh/mlops-zoomcamp.pem
    # Connecting to a service on an internal network from the outside, static forward
    # or set port user forward via on vscode
    LocalForward 8888 localhost:8888
    StrictHostKeyChecking no
```
  
#### Cloud Step 2.6: Connect Created EC2 Instance

```sh  
# Connect Created EC2 Instance
ssh mlops-zoomcamp
```
```sh 
# Check USER is Connected EC2
whoami
```
```sh 
# for exit from AWS EC2 
logout
```



## Environment Steps:  


### Step 0: (İf Required) Install Linux on Windows with WSL

**Installing WSL**: is now easier than ever. Search for **Windows PowerShell** in your Windows search bar, then select **Run as administrator**. 

- [Setting up WSL for a Seamless Data Science Workflow on Windows](https://www.youtube.com/watch?v=IWfsbOzQgXA)

- https://learn.microsoft.com/tr-tr/windows/wsl/install

- https://learn.microsoft.com/tr-tr/windows/wsl/basic-commands


###  Option 1
- At the command prompt type below code, And wait for the process to complete. Ubuntu will then install on your machine. 
```sh 
# Setup WSL install defaulth Ubuntu(WSL)
wsl --install
```
```sh
# Check WSL version
wsl --version
```
```sh
# Update WSL 
wsl --update
```

- **Download Spesific Linux Distros (WSL) on Windows**
```sh 
# Check Available WSL Distros
wsl --list --online
```

###  Option 2 The one line install!

- There is a single command that will install both WSL and Ubuntu at the same time.
When opening PowerShell for the first time, simply modify the initial instruction to:
```sh 
# Download Ubuntu
wsl --install -d ubuntu
```
```sh 
# Set Defaulth WSL Ubuntu
wsl --setdefault ubuntu
```

**Update Ubuntu existing local update packages and upgrade Linux Last Version currently like 23.04 Ubuntu(WSL) on Windows**

- Update existing packages (Required)
```sh
sudo apt update -y
```

- Upgrade Linux currently latest Version (Optionally)
```sh
sudo apt upgrade -y
```
```sh 
# Exit WSL
logout
```

- **Check State and Version Ubuntu(WSL) on Windows**
```sh 
# Check State and Version
wsl -l -v
```
```sh
# if ubuntu version 1 set 2 
wsl --set-version ubuntu 2
```
```sh 
# List Installed WSL Distros
wsl --list --all
```

- **Connect Ubuntu(WSL) on `Windows Powershell` or `Command Prompt (CMD)`**
```sh 
# Connect WSL Ubuntu
wsl
```
```sh 
# Restart or Reconnect All Distros WSL
wsl.exe --shutdown
```

- **Remove Ubuntu(WSL) on Windows**
```sh 
# Remove or Delete Ubuntu
wsl --unregister ubuntu
```

### **Start WSL Ubuntu in specific or current folder on Windows**
- Option 1 cd change directory before wsl command in folder
```sh 
cd C:\Users\<USERNAME>
wsl
```

- Option 2 cd change directory with wsl command in folder
```sh 
wsl --cd C:\Users\<USERNAME>
```

- Option 3 cd change directory after wsl command in folder
```sh 
wsl ~
cd /mnt/c/Users/<USERNAME>
```


### Step 1: Anaconda Distribution `Download` and `Install`.

Open Terminal `Local Mac, Linux, WSL on Windows` or `Connect Cloud Aws EC2 Linux` via SSH as mentioned above.
- https://www.anaconda.com/download#downloads

```sh
# Download Anaconda via wget or curl
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
```
```sh 
# Setup - Attention make sure after installing type "yes" for AUTO connect conda enviroment 
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```
```sh 
# exit AWS EC2 or mac, linux, WSL on Windows
logout
```
```sh 
# Reopen Terminal or Reconnect Cloud and check ACTIVE environment conda base 
ssh mlops-zoomcamp
```


### Step 2: Update existing packages

```sh
sudo apt update -y
```


### Step 3: Install Docker
- https://www.simplilearn.com/tutorials/docker-tutorial
- https://www.simplilearn.com/tutorials/docker-tutorial/how-to-install-docker-on-ubuntu
- https://docs.docker.com/engine/install/linux-postinstall/

**Note**: You directly download the Docker desktop from official site and toggle a key in the docker desktop app to start using Docker in WSL.
- https://docs.docker.com/desktop/windows/wsl/

```sh
sudo apt install docker.io
```

To run docker without `sudo`:

```sh
sudo groupadd docker
sudo usermod -aG docker $USER
```


### Step 4: Install Docker Compose
- https://www.simplilearn.com/tutorials/docker-tutorial/docker-compose#GoTop

Install docker-compose in a separate directory

```sh
mkdir -p ~/soft && cd ~/soft
```

To get the latest release of Docker Compose, go to https://github.com/docker/compose and download the release for your OS.

```sh
wget -N -P "." https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-linux-x86_64 -O docker-compose -q
```

Make it executable

```sh
chmod +x docker-compose
```

Add to the `soft` directory to `PATH`. Open the `.bashrc` file with `nano`:

```sh
nano ~/.bashrc
```

In `.bashrc`, add the following line:

```bash
export PATH="${HOME}/soft:${PATH}"
```

Save it and run the following to make sure the changes are applied:

```bash
source ~/.bashrc
```


### Step 5: Run Docker

**Note**: If you get `docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.` error, restart your VM instance. 

```sh
docker run hello-world
```


### Step 6: Run jupyter notebook

- **Ready For Any Project**

```sh
# go to wsl home 
cd
```
```sh 
# for WSL - go to local HOME windows
cd /mnt/c/Users/your-windows-username
```
```sh 
# create folder jupyter_notebook in HOME folder, then go
mkdir -p jupyter_notebook && cd jupyter_notebook
```
```sh 
# new conda virtual environment
conda create --name tf_py310 python=3.10 jupyter
conda activate tf_py310
```
```sh 
# run jupyter notebook, work on `venv` tf_py310 or base
jupyter notebook
```
```sh 
# Click to any Link access the notebook on Terminal with Ctrl+
# for WSL (if Required)- Close Tilix: Terminal 
```


### Step 7: Open new Notebook with Python Kernel

```sh
# Python Version Check
!python -V
```
```sh 
# Python path check
!which python
```


### Step 8: Setup tensorflow on CPU
```sh
sudo apt update -y
```
```sh 
# EC2 install Tensorflow 200+ mb
pip install tensorflow-cpu
```
```sh 
# or 500+ mb
pip install tensorflow --no-cache-dir
```
```sh 
# for some model flow chart
conda install -c anaconda graphviz pydot
pip install graphviz pydot
sudo apt install graphviz
brew install graphviz
```


### Step 9: Conda conflict, inconsistent or Update

**if cleaning is required**
```sh 
# Attention!, remove unused packages and clear cache, can be remove spme Useful packegec need to install them
conda clean --all
```
```sh 
# collection of pre-installed packages and tools
conda install anaconda --force-reinstall
```

**if The environment is inconsistent**
```sh 
# used to update or install Conda itself
conda update -n base -c defaults conda --force-reinstall
```

**if packages conflict**
```sh 
# check and repair packages conflict
conda update --all --force-reinstall
```


## Useful Scripts

### Install or update the AWS CLI
- https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- https://linuxhint.com/install_aws_cli_ubuntu/

#### Download AWS CLI
```sh
curl -O --create-dirs --output-dir "$HOME/" "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -s
```

#### Install AWS CLI
```sh
sudo apt install unzip -y
unzip ~/awscli-exe-linux-x86_64.zip -d ~/
sudo ~/aws/install
```

#### Configuration AWS CLI
```sh
aws configure
# IAM > Security credentials > Create access key
# Use access keys to send programmatic calls to AWS from the AWS CLI
# AWS Access Key ID [None]	: Access key
# AWS Secret Access Key [None]	: Secret access key
# Default region name [None]	: us-east-1
# Default output format [None]	: text
```

### AWS EC2 IP Updater for SSH Configuration

**Note**: This script automatically updates the IP address of your EC2 instance in your SSH config file. This can be useful if your instance's IP address changes frequently (for example, when stopping and starting an EC2 instance).

```sh
# open and save the following bash script for AWS EC2 ip update:
nano ~/.ssh/update_ssh_config.sh
```

Save the following bash script in a file named `update_ssh_config.sh`, replacing `your_instance_id_here` with your actual AWS EC2 instance ID:

```sh
#!/bin/bash
INSTANCE_ID=your_instance_id_here
REGION=us-east-1
OUTPUT=text

# Get the new public IP address of the EC2 instance
NEW_IP=$(aws ec2 describe-instances  \
                --instance-ids $INSTANCE_ID  \
                --region $REGION \
                --output $OUTPUT \
                --query 'Reservations[0].Instances[0].PublicIpAddress')

# Define the config template
read -r -d '' SSH_CONFIG << EOM

# Updated Configuration in local .ssh/config
# ssh connection calling name
Host mlops-zoomcamp
    # username AWS EC2
    User ubuntu
    # Public IP, it changes when Source EC2 is turned off.
    HostName $NEW_IP
    # Private SSH key file path
    IdentityFile ~/.ssh/mlops-zoomcamp.pem
    # Connecting to a service on an internal network from the outside, static forward
    # or set port user forward via on vscode
    LocalForward 8888 localhost:8888
    StrictHostKeyChecking no
EOM

# Write the new SSH config
echo "$SSH_CONFIG" > ~/.ssh/config
echo "Updated IP: $NEW_IP"
```

```sh
# Make the script executable
chmod +x ~/.ssh/update_ssh_config.sh
```

```sh
# Run the script update: AWS EC2 IP Adress
~/.ssh/update_ssh_config.sh
```

```sh  
# Connect EC2 Instance Updated IP SSH Config
ssh mlops-zoomcamp
```


### Notes
- for Some Errors on WSL
    - [**(OLd Windows) if some error linux-kernel-update-package Ubuntu(WSL) on Windows**](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)

    - [enabling gpu acceleration on ubuntu on wsl2 on Windows](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#2-install-the-appropriate-windows-vgpu-driver-for-wsl)

- GPU CUDA
    - https://developer.nvidia.com/cuda-downloads?target_os=Linux
    - https://lebaohiep.com/shaare/wEd-8g
    
    - **For Ubuntu(WSL)**

        - [WSL2 Installation Failing Miserably](https://discuss.tensorflow.org/t/wsl2-installation-failing-miserably/16236/6)
        - [Tensorflow/Pytorch with CUDA on WSL](https://lebaohiep.com/shaare/wEd-8g)


**if Required but mostly not, GLOBAL setup**

**Note**: It's worth noting that when **using conda**, the CUDA packages are typically sourced from the **Anaconda repository** and might not always have the latest version available. If you require the **latest CUDA version**, using **apt or manually** downloading and installing CUDA from the official NVIDIA website might be necessary.

- https://www.pugetsystems.com/labs/hpc/how-to-install-tensorflow-with-gpu-support-on-windows-10-without-installing-cuda-updated-1419/
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```sh
sudo apt install nvidia-cuda-toolkit
```
```sh
sudo apt install nvidia-cudnn
```

[Building and Scaling a Machine Learning Platform - Magdalena Kuhn](https://www.youtube.com/watch?v=bNrBJwiLBWU)


### **Bash multi Command CheatSheet:**
```sh
"A &"               # Run A in background.
"A ; B"             # Run A and then B, regardless of success of A
"A & B"             # Run A in background and then B, regardless of success of A
"A && B"            # Run B if A succeeded
"A || B"            # Run B if A failed
```

### Next