#!/bin/bash
echo "Remember to disable docker remove in host 4,5 and manually delete running containers."
REMOTE_USER="ubuntu"
REMOTE_IP1="192.168.120.151"
REMOTE_IP2="192.168.120.91"
REMOTE_IP3="192.168.120.79"
REMOTE_IP4="192.168.120.134"
REMOTE_IP5="192.168.120.234"

# Check for help argument
if [ "$1" == "-h" ]; then
    echo "1: cic"
    echo "2: edge"
    echo "3: mnist_3host"
    echo "4: mnist_custom"
    echo "5: mnist_lucid"
    echo "6: mnist_multimodal_2hosts"
    echo "7: mnist_transformer"
    echo "8: edge_multimodal"
    echo "9: edge_multimodal_3hosts"
    echo "10: kltn_multimodal_5hosts"
    exit 0
fi
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm -v -f $(docker ps -qa)"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker container prune -f"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace host{1,2,3,4,5}"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3} swop{1,2,3} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3}-net"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.1 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP1} "pwd && rm -rf SwarmLearning-Setup"
ssh ${REMOTE_USER}@${REMOTE_IP1} "docker ps -a"

ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm -v -f $(docker ps -qa)"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker container prune -f"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace host{1,2,3,4,5}"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3} swop{1,2,3} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3}-net"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.1 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP2} "rm -rf SwarmLearning-Setup"
ssh ${REMOTE_USER}@${REMOTE_IP2} "docker ps -a"

ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker rm -v -f $(docker ps -qa)"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker container prune -f"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3} ."
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace host{1,2,3,4,5}"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3} swop{1,2,3} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3}-net"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.1 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP3} "rm -rf SwarmLearning-Setup"
ssh ${REMOTE_USER}@${REMOTE_IP3} "docker ps -a"

# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker rm -v -f $(docker ps -qa)"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker container prune -f"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3,4,5} ."
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3,4,5} ."
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace host{1,2,3,4,5}"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3,4,5} swop{1,2,3,4,5} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3,4,5}-net"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.1 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP4} "rm -rf SwarmLearning-Setup"
ssh ${REMOTE_USER}@${REMOTE_IP4} "docker ps -a"

# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker rm -v -f $(docker ps -qa)"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker container prune -f"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-scratch{1,2,3,4,5} ."
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && mv workspace/mnist/data-and-edge{1,2,3,4,5} ."
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && rm -rf logs/ workspace host{1,2,3,4,5}"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker rm -f sn{1,2,3,4,5} swop{1,2,3,4,5} swci1"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker rm -f $(docker ps -a | grep 'user-env\|/sl:2' | awk '{print $1}')"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker network rm host-{1,2,3,4,5}-net"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker rm helper"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker image rm tensorflow/tensorflow:2.7.1 user-env-tf2.7.0-swop:latest hello-world:latest"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && docker volume rm $(docker volume ls -q | grep swop)"
ssh ${REMOTE_USER}@${REMOTE_IP5} "rm -rf SwarmLearning-Setup"
ssh ${REMOTE_USER}@${REMOTE_IP5} "docker ps -a"

ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 1"

ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 2"

ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 3"

ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 4"

ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && mkdir workspace && cp -r examples/mnist workspace/ && cp -r examples/utils/gen-cert workspace/mnist/ && chmod 777 -R workspace/"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && ./workspace/mnist/gen-cert -e mnist -i 5"

APLS_IP="apls.uitiot.vn"
SN_1_IP="192.168.120.151"
SN_2_IP="192.168.120.91"
SN_3_IP="192.168.120.79"
SN_4_IP="192.168.120.134"
SN_5_IP="192.168.120.234"
HOST_1_IP="192.168.120.151"
HOST_2_IP="192.168.120.91"
HOST_3_IP="192.168.120.79"
HOST_4_IP="192.168.120.134"
HOST_5_IP="192.168.120.234"
SN_API_PORT="30304"
SN_P2P_PORT="30303"

# Host 1:
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && scp $HOST_2_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-2-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && scp $HOST_3_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-3-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && scp $HOST_4_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-4-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning && scp $HOST_5_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-5-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP1} "git clone https://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@github.com/PNg-HA/SwarmLearning-Setup.git"
# Host 2:
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && scp $HOST_1_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-1-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && scp $HOST_3_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-3-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && scp $HOST_4_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-4-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning && scp $HOST_5_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-5-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP2} "git clone https://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@github.com/PNg-HA/SwarmLearning-Setup.git"
# Host 3:
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && scp $HOST_1_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-1-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && scp $HOST_2_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-2-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && scp $HOST_4_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-4-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning && scp $HOST_5_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-5-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP3} "git clone https://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@github.com/PNg-HA/SwarmLearning-Setup.git"
# Host 4:
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && scp $HOST_1_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-1-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && scp $HOST_2_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-2-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && scp $HOST_3_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-3-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning && scp $HOST_5_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-5-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP4} "git clone https://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@github.com/PNg-HA/SwarmLearning-Setup.git"
# Host 3:
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && scp $HOST_1_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-1-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && scp $HOST_2_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-2-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && scp $HOST_3_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-3-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning && scp $HOST_4_IP:/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath/ca-4-cert.pem /opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath"
ssh ${REMOTE_USER}@${REMOTE_IP5} "git clone https://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@github.com/PNg-HA/SwarmLearning-Setup.git"


case "$1" in
    scenario1)
        # Scenario 1
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/cic && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/cic && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    scenario2)
        # Scenario 2
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/edge && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/edge && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    scenario3)
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
    5)
        # Scenario 5
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/mnist_lucid && cp -r model swop swci data-and-edge1 /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/mnist_lucid && cp -r model swop swci data-and-edge2 /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    6)
        # Scenario 6
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/mnist_lucid && cp -r model swop swci dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/mnist_lucid && cp -r model swop swci dataset /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    7)
        # Scenario 7
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/mnist_multimodal_2hosts && cp -r model swop swci dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/mnist_multimodal_2hosts && cp -r model swop swci dataset /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    8)
        # Scenario 8
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/edge_multimodal_2hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/edge_multimodal_2hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"
        ;;    
    9)
        # Scenario 9
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/edge_multimodal_3hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/edge_multimodal_3hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"
        
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd SwarmLearning-Setup/edge_multimodal_3hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"
        ;;
    10)
        # Scenario 10
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd SwarmLearning-Setup/kltn_multimodal_5hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd SwarmLearning-Setup/kltn_multimodal_5hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"
        
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd SwarmLearning-Setup/kltn_multimodal_5hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP4} "cd SwarmLearning-Setup/kltn_multimodal_5hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"

        ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/workspace/mnist && rm -r model"
        ssh ${REMOTE_USER}@${REMOTE_IP5} "cd SwarmLearning-Setup/kltn_multimodal_5hosts && cp -r model swop swci /opt/hpe/swarm-learning/workspace/mnist/"
        ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && cp -r dataset /opt/hpe/swarm-learning/workspace/mnist/"
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
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP1} "docker network create host-1-net"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
ssh ${REMOTE_USER}@${REMOTE_IP1} "docker rm helper"


ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_2_IP}+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP2} "docker network create host-2-net"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
ssh ${REMOTE_USER}@${REMOTE_IP2} "docker rm helper"

ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-3-net+g\" workspace/mnist/swop/swop3_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_3_IP}+g\" workspace/mnist/swop/swop3_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
ssh ${REMOTE_USER}@${REMOTE_IP3} "docker network create host-3-net"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
ssh ${REMOTE_USER}@${REMOTE_IP3} "docker rm helper"

# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-4-net+g\" workspace/mnist/swop/swop4_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_4_IP}+g\" workspace/mnist/swop/swop4_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "docker network create host-4-net"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "docker rm helper"

# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-MODEL>+/opt/hpe/swarm-learning/workspace/mnist/model+g\" workspace/mnist/swci/taskdefs/swarm_mnist_task.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-1-net+g\" workspace/mnist/swop/swop1_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-2-net+g\" workspace/mnist/swop/swop2_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<SWARM-NETWORK>+host-5-net+g\" workspace/mnist/swop/swop5_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_1_IP}+g\" workspace/mnist/swop/swop1_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<HOST_ADDRESS>+${HOST_5_IP}+g\" workspace/mnist/swop/swop5_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT>+/opt/hpe/swarm-learning/workspace/mnist+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sed -i \"s+<PROJECT-CACERTS>+/opt/hpe/swarm-learning/workspace/mnist/cert/ca/capath+g\" workspace/mnist/swop/swop*_profile.yaml"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "docker network create host-5-net"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && docker volume rm sl-cli-lib"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && docker volume create sl-cli-lib; docker container create --name helper -v sl-cli-lib:/data hello-world"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "docker rm helper"


echo "sn1 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn1 \
--network=host-1-net --host-ip=${HOST_1_IP} \
--sentinel --sn-p2p-port=${SN_P2P_PORT} \
--sn-api-port=${SN_API_PORT} \
--key=workspace/mnist/cert/sn-1-key.pem \
--cert=workspace/mnist/cert/sn-1-cert.pem \
--capath=workspace/mnist/cert/ca/capath \
--apls-ip=${APLS_IP} \
--apls-port=443"


sleep 210
SEARCH_PHRASE="Starting SWARM-API-SERVER on port"
# Function to check for the phrase in the docker logs
check_logs1() {
    ssh ${REMOTE_USER}@${REMOTE_IP1} "docker logs sn1" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
    return 1
}

# Loop until the phrase is found
while true; do
    if check_logs1; then
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
--apls-ip=${APLS_IP} \
--apls-port=443"



sleep 90
# Function to check for the phrase in the docker logs
check_logs2() {
    ssh ${REMOTE_USER}@${REMOTE_IP2} "docker logs sn2" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
    return 1
}

while true; do
    if check_logs2; then
        echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
        break
    else
        echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
        sleep 5
    fi
done

echo "sn3 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn3 \
--network=host-3-net --host-ip=${HOST_3_IP} \
--sentinel-ip=${SN_1_IP} --sn-p2p-port=${SN_P2P_PORT} \
--sn-api-port=${SN_API_PORT} --key=workspace/mnist/cert/sn-3-key.pem \
--cert=workspace/mnist/cert/sn-3-cert.pem \
--capath=workspace/mnist/cert/ca/capath \
--apls-ip=${APLS_IP} \
--apls-port=443"


sleep 90
# Function to check for the phrase in the docker logs
check_logs3() {
    ssh ${REMOTE_USER}@${REMOTE_IP3} "docker logs sn3" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
    return 1
}

# Loop until the phrase is found
while true; do
    if check_logs3; then
        echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
        break
    else
        echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
        sleep 5
    fi
done


# echo "sn4 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn4 \
# --network=host-4-net --host-ip=${HOST_4_IP} \
# --sentinel-ip=${SN_1_IP} --sn-p2p-port=${SN_P2P_PORT} \
# --sn-api-port=${SN_API_PORT} --key=workspace/mnist/cert/sn-4-key.pem \
# --cert=workspace/mnist/cert/sn-4-cert.pem \
# --capath=workspace/mnist/cert/ca/capath \
# --apls-ip=${APLS_IP} \
# --apls-port=443"


# sleep 90
# # Function to check for the phrase in the docker logs
# check_logs4() {
#     ssh ${REMOTE_USER}@${REMOTE_IP4} "docker logs sn4" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
#     return 1
# }

# # Loop until the phrase is found
# while true; do
#     if check_logs4; then
#         echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
#         break
#     else
#         echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
#         sleep 5
#     fi
# done

# echo "sn5 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-sn -d --rm --name=sn5 \
# --network=host-5-net --host-ip=${HOST_5_IP} \
# --sentinel-ip=${SN_1_IP} --sn-p2p-port=${SN_P2P_PORT} \
# --sn-api-port=${SN_API_PORT} --key=workspace/mnist/cert/sn-5-key.pem \
# --cert=workspace/mnist/cert/sn-5-cert.pem \
# --capath=workspace/mnist/cert/ca/capath \
# --apls-ip=${APLS_IP} \
# --apls-port=443"


# sleep 90
# # Function to check for the phrase in the docker logs
# check_logs5() {
#     ssh ${REMOTE_USER}@${REMOTE_IP5} "docker logs sn5" | grep --line-buffered "${SEARCH_PHRASE}" && return 0
#     return 1
# }

# # Loop until the phrase is found
# while true; do
#     if check_logs5; then
#         echo "Phrase '${SEARCH_PHRASE}' found. Continuing script..."
#         break
#     else
#         echo "Phrase '${SEARCH_PHRASE}' not found yet. Sleeping for 5 seconds..."
#         sleep 5
#     fi
# done


APLS_IP="apls.uitiot.vn"
SN_1_IP="192.168.120.151"
SN_2_IP="192.168.120.91"
SN_3_IP="192.168.120.79"
SN_4_IP="192.168.120.134"
SN_5_IP="192.168.120.234"
HOST_1_IP="192.168.120.151"
HOST_2_IP="192.168.120.91"
HOST_3_IP="192.168.120.79"
HOST_4_IP="192.168.120.134"
HOST_5_IP="192.168.120.234"
SN_API_PORT="30304"
SN_P2P_PORT="30303"


echo "swop1 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop1 --network=host-1-net \
--sn-ip=${SN_1_IP} --sn-api-port=${SN_API_PORT} \
--usr-dir=workspace/mnist/swop --profile-file-name=swop1_profile.yaml \
--key=workspace/mnist/cert/swop-1-key.pem \
--cert=workspace/mnist/cert/swop-1-cert.pem \
--capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
--apls-ip=${APLS_IP} \
--apls-port=443"

echo "swop2 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop2 --network=host-2-net \
--sn-ip=${SN_2_IP} --sn-api-port=${SN_API_PORT} \
--usr-dir=workspace/mnist/swop --profile-file-name=swop2_profile.yaml \
--key=workspace/mnist/cert/swop-2-key.pem \
--cert=workspace/mnist/cert/swop-2-cert.pem \
--capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
--apls-ip=${APLS_IP} \
--apls-port=443"

echo "swop3 start: ..."
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop3 --network=host-3-net \
--sn-ip=${SN_3_IP} --sn-api-port=${SN_API_PORT} \
--usr-dir=workspace/mnist/swop --profile-file-name=swop3_profile.yaml \
--key=workspace/mnist/cert/swop-3-key.pem \
--cert=workspace/mnist/cert/swop-3-cert.pem \
--capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
--apls-ip=${APLS_IP} \
--apls-port=443"

# echo "swop4 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop4 --network=host-4-net \
# --sn-ip=${SN_4_IP} --sn-api-port=${SN_API_PORT} \
# --usr-dir=workspace/mnist/swop --profile-file-name=swop4_profile.yaml \
# --key=workspace/mnist/cert/swop-4-key.pem \
# --cert=workspace/mnist/cert/swop-4-cert.pem \
# --capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
# --apls-ip=${APLS_IP} \
# --apls-port=443"

# echo "swop5 start: ..."
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swop -d --name=swop5 --network=host-5-net \
# --sn-ip=${SN_5_IP} --sn-api-port=${SN_API_PORT} \
# --usr-dir=workspace/mnist/swop --profile-file-name=swop5_profile.yaml \
# --key=workspace/mnist/cert/swop-5-key.pem \
# --cert=workspace/mnist/cert/swop-5-cert.pem \
# --capath=workspace/mnist/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= \
# --apls-ip=${APLS_IP} \
# --apls-port=443"


ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && ./scripts/bin/run-swci --name=swci1 --network=host-1-net \
--usr-dir=workspace/mnist/swci --init-script-name=swci-init \
--key=workspace/mnist/cert/swci-1-key.pem \
--cert=workspace/mnist/cert/swci-1-cert.pem \
--capath=workspace/mnist/cert/ca/capath \
-e http_proxy= -e https_proxy= --apls-ip=${APLS_IP} --apls-port=443"
# ssh ${REMOTE_USER}@${REMOTE_IP2} ""

## ./scripts/bin/run-swci --name=swci1 --network=host-1-net \
## --usr-dir=workspace/mnist/swci --init-script-name=swci-init \
## --key=workspace/mnist/cert/swci-1-key.pem \
## --cert=workspace/mnist/cert/swci-1-cert.pem \
## --capath=workspace/mnist/cert/ca/capath \
## -e http_proxy= -e https_proxy= --apls-ip=${APLS_IP} --apls-port=443


ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && sudo ./scripts/bin/swarmLogCollector "hub.myenterpriselicense.hpe.com/hpe/swarm-learning" "workspace=/opt/hpe/swarm-learning/workspace/mnist" && sudo mv /opt/logs/*swarm_log* ."
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && sudo ./scripts/bin/swarmLogCollector "hub.myenterpriselicense.hpe.com/hpe/swarm-learning" "workspace=/opt/hpe/swarm-learning/workspace/mnist" && sudo mv /opt/logs/*swarm_log* ."
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && sudo ./scripts/bin/swarmLogCollector "hub.myenterpriselicense.hpe.com/hpe/swarm-learning" "workspace=/opt/hpe/swarm-learning/workspace/mnist" && sudo mv /opt/logs/*swarm_log* ."
ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && sudo ./scripts/bin/swarmLogCollector "hub.myenterpriselicense.hpe.com/hpe/swarm-learning" "workspace=/opt/hpe/swarm-learning/workspace/mnist" && sudo mv /opt/logs/*swarm_log* ."
ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && sudo ./scripts/bin/swarmLogCollector "hub.myenterpriselicense.hpe.com/hpe/swarm-learning" "workspace=/opt/hpe/swarm-learning/workspace/mnist" && sudo mv /opt/logs/*swarm_log* ."

ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ "
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && echo 'loss,accuracy' > ml1.csv"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && docker logs $(docker ps -a | grep demo-swarm_mnist_task-u-0 | awk '{print $1}') | grep val_loss | awk '{print $8\",\"$11}' >> ml1.csv"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && mkdir host1; mv ml1.csv ml1.txt swarm_logs_* memory_usage.csv host1"
ssh ${REMOTE_USER}@${REMOTE_IP1} "cd /opt/hpe/swarm-learning/ && cp -r workspace/mnist/dataset/user1/ host1"

ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ "
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && echo 'loss,accuracy' > ml2.csv"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && docker logs $(docker ps -a | grep demo-swarm_mnist_task-u-0 | awk '{print $1}') | grep val_loss | awk '{print $8\",\"$11}' >> ml2.csv"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && mkdir host2; mv ml2.csv ml2.txt swarm_logs_* memory_usage.csv host2"
ssh ${REMOTE_USER}@${REMOTE_IP2} "cd /opt/hpe/swarm-learning/ && cp -r workspace/mnist/dataset/user2/ host2"

ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ "
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && echo 'loss,accuracy' > ml3.csv"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && docker logs $(docker ps -a | grep demo-swarm_mnist_task-u-0 | awk '{print $1}') | grep val_loss | awk '{print $8\",\"$11}' >> ml3.csv"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && mkdir host3; mv ml3.csv ml3.txt swarm_logs_* memory_usage.csv host3"
ssh ${REMOTE_USER}@${REMOTE_IP3} "cd /opt/hpe/swarm-learning/ && cp -r workspace/mnist/dataset/user3/ host3"

# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ "
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && echo 'loss,accuracy' > ml4.csv"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && docker logs $(docker ps -a | grep demo-swarm_mnist_task-u-0 | awk '{print $1}') | grep val_loss | awk '{print $8\",\"$11}' >> ml4.csv"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && mkdir host4; mv ml4.csv ml4.txt swarm_logs_* memory_usage.csv host4"
# ssh ${REMOTE_USER}@${REMOTE_IP4} "cd /opt/hpe/swarm-learning/ && cp -r workspace/mnist/dataset/user4/ host4"

# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ "
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && echo 'loss,accuracy' > ml5.csv"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && docker logs $(docker ps -a | grep demo-swarm_mnist_task-u-0 | awk '{print $1}') | grep val_loss | awk '{print $8\",\"$11}' >> ml5.csv"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && mkdir host5; mv ml5.csv ml5.txt swarm_logs_* memory_usage.csv host5"
# ssh ${REMOTE_USER}@${REMOTE_IP5} "cd /opt/hpe/swarm-learning/ && cp -r workspace/mnist/dataset/user5/ host5"