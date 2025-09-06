pipeline {
  agent any
  environment {
    AWS_REGION = 'us-east-1'
    REPO = 'clf-onnx-api'
  }
  options { timestamps() }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Setup Python') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install sagemaker boto3 onnxruntime pytest ruff black
        '''
      }
    }

    stage('AWS Login & ECR Prep') {
      environment {
        AWS_ACCESS_KEY_ID     = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
      }
      steps {
        sh '''
          set -e
          ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
          ECR=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
          aws ecr describe-repositories --repository-name $REPO --region $AWS_REGION || \
            aws ecr create-repository --repository-name $REPO --region $AWS_REGION
          aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR
        '''
      }
    }

    stage('Build & Push Docker') {
      steps {
        withCredentials([
          string(credentialsId: 'aws-access-key',  variable: 'AWS_ACCESS_KEY_ID'),
          string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY')
        ]) {
          sh '''
            set -e
            TAG=smv${BUILD_NUMBER}
            ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
            ECR=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

            aws ecr describe-repositories --repository-name $REPO --region $AWS_REGION || \
              aws ecr create-repository --repository-name $REPO --region $AWS_REGION
            aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR

            docker build -t $REPO:$TAG .
            docker tag $REPO:$TAG $ECR/$REPO:$TAG
            docker push $ECR/$REPO:$TAG

            echo ACCOUNT_ID=$ACCOUNT_ID > env.out
            echo AWS_REGION=$AWS_REGION >> env.out
            echo ECR=$ECR >> env.out
            echo IMAGE_URI=$ECR/$REPO:$TAG >> env.out
          '''
        }
      }
    }

    stage('Package & Upload Model') {
        steps {
            withCredentials([
            string(credentialsId: 'aws-access-key',  variable: 'AWS_ACCESS_KEY_ID'),
            string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY')
            ]) {
            sh '''
                set -e
                AWS_REGION=us-east-1
                ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
                BUCKET="clf-artifacts-$ACCOUNT_ID-$AWS_REGION"

                aws s3 mb s3://$BUCKET --region $AWS_REGION || true
                mkdir -p model_art
                # get ONNX (either present locally or pull a cached copy)
                if [ ! -f model_int8_qdq.onnx ]; then
                aws s3 cp s3://$BUCKET/models/model_int8_qdq.onnx model_int8_qdq.onnx || true
                fi
                cp model_int8_qdq.onnx model_art/
                (cd model_art && tar -czf model.tar.gz model_int8_qdq.onnx)
                aws s3 cp model_art/model.tar.gz s3://$BUCKET/models/model.tar.gz --region $AWS_REGION

                # write env.out for later stage (optional)
                echo ACCOUNT_ID=$ACCOUNT_ID > env.out
                echo AWS_REGION=$AWS_REGION >> env.out
                echo MODEL_S3=s3://$BUCKET/models/model.tar.gz >> env.out
            '''
            }
        }
    }
stage('Deploy SageMaker') {
  environment { SAGEMAKER_ROLE = credentials('sagemaker-exec-role-arn') }
  steps {
    withCredentials([
      string(credentialsId: 'aws-access-key',  variable: 'AWS_ACCESS_KEY_ID'),
      string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY')
    ]) {
      sh '''
        set -e
        AWS_REGION=us-east-1
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        REPO=clf-onnx-api
        # get latest image URI (last pushed)
        IMAGE_URI=$(aws ecr describe-images --repository-name $REPO --region $AWS_REGION \
          --query "reverse(sort_by(imageDetails,& imagePushedAt))[0].imageUri" --output text)
        MODEL_S3=s3://clf-artifacts-$ACCOUNT_ID-$AWS_REGION/models/model.tar.gz

        . .venv/bin/activate
        python - <<'PY'
import os, sagemaker
from sagemaker.model import Model
img=os.environ['IMAGE_URI']
mdata=os.environ['MODEL_S3']
role=os.environ['SAGEMAKER_ROLE']
sess=sagemaker.Session()
Model(image_uri=img, role=role, model_data=mdata,
      env={"MODEL_PATH":"/opt/ml/model/model_int8_qdq.onnx"},
      sagemaker_session=sess).deploy(
    endpoint_name="clf-onnx-endpoint1",
    instance_type="ml.t2.medium",
    initial_instance_count=1)
print("Deployed.")
PY
      '''
    }
  }
}
  }

  post {
    always {
      archiveArtifacts artifacts: 'env.out, model_art/model.tar.gz', onlyIfSuccessful: false
    }
  }
}