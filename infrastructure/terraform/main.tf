# ============================================================================
# MLOps Infrastructure - AWS Terraform Configuration
# ============================================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ============================================================================
# Local Variables
# ============================================================================

locals {
  project_name = "mlops-hassan"
  common_tags = {
    Project     = "MLOps Infrastructure"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ============================================================================
# Random suffix for unique names
# ============================================================================

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "random_password" "api_key" {
  length  = 32
  special = false
}

# ============================================================================
# Service 1: IAM - Roles and Policies
# ============================================================================

resource "aws_iam_role" "lambda_execution" {
  name = "${local.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${local.project_name}-lambda-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlops_artifacts.arn,
          "${aws_s3_bucket.mlops_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
          "dynamodb:DeleteItem", "dynamodb:Scan", "dynamodb:Query"
        ]
        Resource = aws_dynamodb_table.mlops_requests.arn
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.mlops_alerts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:*:parameter/mlops/*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = aws_ecr_repository.mlops_api.arn
      },
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ============================================================================
# Service 2: S3 - Artifact Storage
# ============================================================================

resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "${local.project_name}-artifacts-${random_string.suffix.result}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_public_access_block" "mlops_artifacts" {
  bucket                  = aws_s3_bucket.mlops_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "mlops_artifacts" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "mlops_artifacts" {
  bucket = aws_s3_bucket.mlops_artifacts.id

  rule {
    id     = "expire-old-models"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "expire-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = 30
    }
  }
}

# ============================================================================
# Service 3: ECR - Container Registry
# ============================================================================

resource "aws_ecr_repository" "mlops_api" {
  name                 = "${local.project_name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "mlops_api" {
  repository = aws_ecr_repository.mlops_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 3 images"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# ============================================================================
# Service 4: Lambda - Serverless ML Inference
# ============================================================================

resource "aws_lambda_function" "mlops_api" {
  function_name = "${local.project_name}-api"
  role          = aws_iam_role.lambda_execution.arn

  package_type = "Image"
  image_uri    = "${aws_ecr_repository.mlops_api.repository_url}:latest"

  memory_size = 512
  timeout     = 60

  environment {
    variables = {
      S3_BUCKET           = aws_s3_bucket.mlops_artifacts.id
      DYNAMODB_TABLE      = aws_dynamodb_table.mlops_requests.name
      SNS_TOPIC_ARN       = aws_sns_topic.mlops_alerts.arn
      MLFLOW_TRACKING_URI = "s3://${aws_s3_bucket.mlops_artifacts.id}/mlflow"
      EXPERIMENT_STORAGE  = "/tmp/experiments.json"
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mlops_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.mlops_api.execution_arn}/*/*"
}

# ============================================================================
# Service 5: DynamoDB - Request Logging
# ============================================================================

resource "aws_dynamodb_table" "mlops_requests" {
  name         = "${local.project_name}-requests"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "request_id"

  attribute {
    name = "request_id"
    type = "S"
  }

  ttl {
    attribute_name = "expiry_time"
    enabled        = true
  }

  tags = local.common_tags
}

# ============================================================================
# Service 6: API Gateway - HTTP API
# ============================================================================

resource "aws_api_gateway_rest_api" "mlops_api" {
  name        = "${local.project_name}-api"
  description = "MLOps API Gateway"
  tags        = local.common_tags
}

resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.mlops_api.id
  parent_id   = aws_api_gateway_rest_api.mlops_api.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  rest_api_id   = aws_api_gateway_rest_api.mlops_api.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "proxy_root" {
  rest_api_id   = aws_api_gateway_rest_api.mlops_api.id
  resource_id   = aws_api_gateway_rest_api.mlops_api.root_resource_id
  http_method   = "ANY"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id             = aws_api_gateway_rest_api.mlops_api.id
  resource_id             = aws_api_gateway_resource.proxy.id
  http_method             = aws_api_gateway_method.proxy.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.mlops_api.invoke_arn
}

resource "aws_api_gateway_integration" "lambda_root" {
  rest_api_id             = aws_api_gateway_rest_api.mlops_api.id
  resource_id             = aws_api_gateway_rest_api.mlops_api.root_resource_id
  http_method             = aws_api_gateway_method.proxy_root.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.mlops_api.invoke_arn
}

resource "aws_api_gateway_deployment" "mlops_api" {
  depends_on = [
    aws_api_gateway_integration.lambda,
    aws_api_gateway_integration.lambda_root
  ]

  rest_api_id = aws_api_gateway_rest_api.mlops_api.id
  stage_name  = var.environment

  lifecycle {
    create_before_destroy = true
  }
}


# ============================================================================
# Service 7: SNS - Alerts
# ============================================================================

resource "aws_sns_topic" "mlops_alerts" {
  name = "${local.project_name}-alerts"
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.mlops_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ============================================================================
# Service 8: CloudWatch - Monitoring
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "api_gateway_5xx" {
  alarm_name          = "${local.project_name}-api-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "5XXError"
  namespace           = "AWS/ApiGateway"
  period              = "120"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "API Gateway 5XX errors"
  alarm_actions       = [aws_sns_topic.mlops_alerts.arn]

  dimensions = {
    ApiName = aws_api_gateway_rest_api.mlops_api.name
  }

  tags = local.common_tags
}

# ============================================================================
# Service 9: SSM Parameter Store - Configuration
# ============================================================================

resource "aws_ssm_parameter" "api_key" {
  name  = "/mlops/api-key"
  type  = "SecureString"
  value = random_password.api_key.result
  tags  = local.common_tags
}

resource "aws_ssm_parameter" "mlflow_tracking_uri" {
  name  = "/mlops/mlflow-tracking-uri"
  type  = "String"
  value = "s3://${aws_s3_bucket.mlops_artifacts.id}/mlflow"
  tags  = local.common_tags
}

# ============================================================================
# EventBridge - Daily Drift Check
# ============================================================================

resource "aws_cloudwatch_event_rule" "daily_drift_check" {
  name                = "${local.project_name}-daily-drift"
  description         = "Trigger daily drift detection"
  schedule_expression = "rate(1 day)"
  tags                = local.common_tags
}

resource "aws_cloudwatch_event_target" "drift_lambda" {
  rule      = aws_cloudwatch_event_rule.daily_drift_check.name
  target_id = "DriftCheckLambda"
  arn       = aws_lambda_function.mlops_api.arn
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mlops_api.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_drift_check.arn
}

# ============================================================================
# Outputs
# ============================================================================

output "api_gateway_url" {
  description = "API Gateway URL"
  value       = "https://${aws_api_gateway_rest_api.mlops_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}"
}

output "s3_bucket_name" {
  description = "S3 bucket for ML artifacts"
  value       = aws_s3_bucket.mlops_artifacts.id
}

output "dynamodb_table_name" {
  description = "DynamoDB table for request logging"
  value       = aws_dynamodb_table.mlops_requests.name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.mlops_api.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.mlops_api.function_name
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.mlops_alerts.arn
}
