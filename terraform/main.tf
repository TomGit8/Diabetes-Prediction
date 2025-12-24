terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" # Région standard pour les comptes étudiants AWS Academy
}

# 1. ECR Repository - Pour stocker les images Docker
resource "aws_ecr_repository" "app_repo" {
  name                 = "ecr-g3mg05"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # Permet de détruire le repo même s'il contient des images (utile pour le dev)

  image_scanning_configuration {
    scan_on_push = true
  }
}

# 2. S3 Bucket - Pour stocker les modèles et artefacts
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "s3-g3mg05"
  force_destroy = true # Permet de détruire le bucket même s'il n'est pas vide
}

# Blocage de l'accès public pour S3 (Sécurité)
resource "aws_s3_bucket_public_access_block" "model_bucket_access" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 3. IAM Role pour App Runner (Pour qu'il puisse pull l'image depuis ECR)
resource "aws_iam_role" "apprunner_role" {
  name = "AppRunnerECRAccessRole-g3mg05"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_policy" {
  role       = aws_iam_role.apprunner_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# 4. App Runner Service - L'hébergement de l'application
# NOTE: Ce service nécessitera qu'une image soit d'abord poussée sur ECR pour démarrer correctement.
resource "aws_apprunner_service" "app_service" {
  service_name = "apprunner-g3mg05"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_role.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.app_repo.repository_url}:latest"
      image_repository_type = "ECR"
      
      image_configuration {
        port = "8501" # Port par défaut de Streamlit
         runtime_environment_variables = {
          PORT = "8501"
        }
      }
    }
    
    auto_deployments_enabled = true
  }

  instance_configuration {
    cpu    = "1024" # 1 vCPU
    memory = "2048" # 2 GB
  }

  depends_on = [aws_iam_role_policy_attachment.apprunner_policy]
}

# Outputs pour faciliter la récupération des infos
output "ecr_repository_url" {
  value = aws_ecr_repository.app_repo.repository_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.model_bucket.id
}

output "app_runner_url" {
  value = aws_apprunner_service.app_service.service_url
}
