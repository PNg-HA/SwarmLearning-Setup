#!/bin/bash

REMOTE_USER="ubuntu"
REMOTE_IP1="192."
REMOTE_IP2="192."
REMOTE_PASSWORD="abc"
REMOTE_COMMAND=""

# Check for help argument
if [ "$1" == "-h" ]; then
    echo "1: cic"
    echo "2: edge"
    echo "3: mnist_3host"
    echo "4: mnist_custom"
    exit 0
fi

ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3} swop{1,2,3} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3}-net"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.0 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP1} "pwd && rm -rf SwarmLearning-Setup"

ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3} swop{1,2,3} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3}-net"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.0 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP2} "rm -rf SwarmLearning-Setup"

ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 1"

ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 2"

APLS_IP="192.168.120.104"
SN_1_IP="192.168.120.104"
SN_2_IP="192.168.120.118"
HOST_1_IP="192.168.120.104"
HOST_2_IP="192.168.120.118"
SN_API_PORT="30304"
SN_P2P_PORT="30303"

# Host 1:
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && scp $HOST_2_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-2-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP1} "git clone https://github.com/PNg-HA/SwarmLearning-Setup.git"
# Host 2:
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && scp $HOST_1_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-1-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP2} "git clone https://github.com/PNg-HA/SwarmLearning-Setup.git"

case "$1" in
    1)
        # Scenario 1
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/cic && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/cic && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    2)
        # Scenario 2
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/edge && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/edge && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    3)
        # Scenario 3
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/mnist_3host && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/mnist_3host && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    4)
        # Scenario 4
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/mnist_custom && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/mnist_custom && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    *)
        echo "Invalid scenario. Use -h for help."
        exit 1
        ;;
esac

ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_2_IP}+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "docker network create host-1-net"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
ssh ${REMOTE_USER}@${REMOTE_IP1} "docker rm helper"
# ssh ${REMOTE_USER}@${REMOTE_IP1} ""
# ssh ${REMOTE_USER}@${REMOTE_IP1} ""




ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_2_IP}+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "docker network create host-2-net"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
ssh ${REMOTE_USER}@${REMOTE_IP2} "docker rm helper"

echo "sn1 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn1 \
--network=host-1-net --host-ip=${HOST_1_IP} \
--sentinel --sn-p2p-port=${SN_P2P_PORT} \
--sn-api-port=${SN_API_PORT} \
--key=workspace/mnist/cert/sn-1-key.pem \
--cert=workspace/mnist/cert/sn-1-cert.pem \
--capath=workspace/mnist/cert/ca/capath \
--apls-ip=${APLS_IP}"


sleep 240
SEARCH_PHRASE="Starting SWARM-API-SERVER on port"
# Function to check for the phrase in the docker logs
check_logs() {
    ssh ${REMOTE_USER}@${REMOTE_IP} "docker logs -f sn1" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
    return 1
}

# Loop until the phrase is found
while true; do
    if check_logs; then
        echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
        break
    else
        echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
        sleep 5
    fi
done

# Continue with the rest of your script
echo "sn2 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn2 \
--network=host-2-net --host-ip=${HOST_2_IP} \
--sentinel-ip=${SN_1_IP} --sn-p2p-port=${SN_P2P_PORT} \
--sn-api-port=${SN_API_PORT} --key=workspace/mnist/cert/sn-2-key.pem \
--cert=workspace/mnist/cert/sn-2-cert.pem \
--capath=workspace/mnist/cert/ca/capath \
--apls-ip=${APLS_IP}"

ssh ${REMOTE_USER}@${REMOTE_IP} "docker logs -f sn2"
sleep 180
# Function to check for the phrase in the docker logs
check_logs() {
    ssh ${REMOTE_USER}@${REMOTE_IP} "docker logs -f sn2" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
    return 1
}

# Loop until the phrase is found
while true; do
    if check_logs; then
        echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
        break
    else
        echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
        sleep 5
    fi
done

# Continue with the rest of your script
# echo "swop1 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop1 --network=host-1-net \
# --sn-ip=${SN_1_IP} --sn-api-port=${SN_API_PORT} \
# --usr-dir=workspace/mnist/swop --profile-file-name=swop1_profile.yaml \
# --key=workspace/mnist/cert/swop-1-key.pem \
# --cert=workspace/mnist/cert/swop-1-cert.pem \
# --capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
# --apls-ip=${APLS_IP}"

# echo "swop2 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop1 --network=host-1-net \
# --sn-ip=${SN_1_IP} --sn-api-port=${SN_API_PORT} \
# --usr-dir=workspace/mnist/swop --profile-file-name=swop1_profile.yaml \
# --key=workspace/mnist/cert/swop-1-key.pem \
# --cert=workspace/mnist/cert/swop-1-cert.pem \
# --capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
# --apls-ip=${APLS_IP}"

# ssh ${REMOTE_USER}@${REMOTE_IP2} ""
