# Values the pre-serve migration step needs (scripts/ecs_migrate.py,
# docs/pilot/runbook.md). Identifiers only; no secrets.

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "api_task_definition_family" {
  value = aws_ecs_task_definition.api.family
}

output "api_task_definition_arn" {
  description = "Revision registered by the last apply; run migrations with it before the service uses it"
  value       = aws_ecs_task_definition.api.arn
}

output "api_service_name" {
  value = aws_ecs_service.api.name
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "api_security_group_id" {
  value = aws_security_group.api.id
}

output "api_log_group" {
  value = aws_cloudwatch_log_group.api.name
}
