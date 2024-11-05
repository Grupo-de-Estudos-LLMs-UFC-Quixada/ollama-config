#!/bin/bash

# Define as variáveis
AMI_ID=$(aws ec2 describe-images \
	--owners 099720109477 \
       	--filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
	"Name=virtualization-type,Values=hvm" \
	"Name=architecture,Values=x86_64" \
	--query 'Images[0].ImageId' \
	--output text )
INSTANCE_TYPE="z1d.large"
VOLUME_SIZE=50
KEY_NAME="vockey" # Substitua pelo nome do seu par de chaves
USER_DATA=$(cat setup.sh | base64 -w 0)

# Cria o grupo de segurança
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name "ollama" \
    --description "Treinamento LLM" \
    --query 'GroupId' \
    --output text)

echo "Grupo de segurança criado com ID: $SECURITY_GROUP_ID"

# Libera as portas 22, 80 e 443
aws ec2 authorize-security-group-ingress \
    --group-id "$SECURITY_GROUP_ID" \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --no-cli-pager

aws ec2 authorize-security-group-ingress \
    --group-id "$SECURITY_GROUP_ID" \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --no-cli-pager

aws ec2 authorize-security-group-ingress \
    --group-id "$SECURITY_GROUP_ID" \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --no-cli-pager

echo "Portas 22, 80 e 443 liberadas no grupo de segurança."

# Cria a instância (mesmo código do script anterior, usando SECURITY_GROUP_ID)
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --block-device-mappings "[{\"DeviceName\": \"/dev/sda1\", \"Ebs\": {\"VolumeSize\": $VOLUME_SIZE, \"VolumeType\": \"gp3\"}}]" \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ollama}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instância criada com ID: $INSTANCE_ID"

# Aguarda a instância estar em execução
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Obtém o IP público da instância
aws ec2 describe-instances \
	--instance-ids "$INSTANCE_ID" \
	--query 'Reservations[0].Instances[0].PublicIpAddress' \
	--output text \
	--no-cli-pager

