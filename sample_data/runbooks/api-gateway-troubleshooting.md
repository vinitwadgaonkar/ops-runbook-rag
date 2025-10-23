---
service: api-gateway
severity: critical
component: ingress
team: platform
tags: [troubleshooting, networking, load-balancing]
---

# API Gateway Troubleshooting Guide

## Overview
This runbook provides step-by-step procedures for troubleshooting API Gateway issues in our production environment.

## Common Issues

### 1. 503 Service Unavailable Errors

**Symptoms:**
- Clients receiving 503 errors
- High error rates in monitoring
- Load balancer health checks failing

**Diagnosis Steps:**

1. **Check upstream service health:**
   ```bash
   # Check if upstream services are healthy
   kubectl get pods -n api-gateway
   kubectl describe service api-gateway-service
   ```

2. **Verify load balancer configuration:**
   ```bash
   # Check ingress controller logs
   kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
   
   # Verify backend services
   kubectl get endpoints api-gateway-service
   ```

3. **Check resource utilization:**
   ```bash
   # Check CPU and memory usage
   kubectl top pods -n api-gateway
   
   # Check for resource limits
   kubectl describe pod <pod-name> -n api-gateway
   ```

**Resolution Steps:**

1. **Scale up if needed:**
   ```bash
   kubectl scale deployment api-gateway --replicas=3 -n api-gateway
   ```

2. **Restart unhealthy pods:**
   ```bash
   kubectl delete pod <unhealthy-pod-name> -n api-gateway
   ```

3. **Check and update ingress configuration:**
   ```bash
   kubectl get ingress api-gateway-ingress -o yaml
   kubectl apply -f api-gateway-ingress.yaml
   ```

### 2. High Latency Issues

**Symptoms:**
- Response times > 2 seconds
- Timeout errors
- Slow database queries

**Diagnosis Steps:**

1. **Check application metrics:**
   ```bash
   # View application logs
   kubectl logs -f deployment/api-gateway -n api-gateway
   
   # Check Prometheus metrics
   curl http://prometheus:9090/api/v1/query?query=api_gateway_request_duration_seconds
   ```

2. **Database connection analysis:**
   ```bash
   # Check database connections
   kubectl exec -it <api-gateway-pod> -- netstat -an | grep :5432
   
   # Check connection pool metrics
   curl http://api-gateway:8080/metrics | grep connection_pool
   ```

**Resolution Steps:**

1. **Optimize database queries:**
   ```bash
   # Enable query logging
   kubectl exec -it <api-gateway-pod> -- psql -c "SET log_statement = 'all';"
   ```

2. **Adjust connection pool settings:**
   ```yaml
   # Update deployment with new environment variables
   env:
     - name: DB_POOL_SIZE
       value: "20"
     - name: DB_POOL_TIMEOUT
       value: "30s"
   ```

### 3. Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- JWT token validation failures
- OAuth integration issues

**Diagnosis Steps:**

1. **Check authentication service:**
   ```bash
   # Verify auth service is running
   kubectl get pods -n auth-service
   
   # Check auth service logs
   kubectl logs -f deployment/auth-service -n auth-service
   ```

2. **Validate JWT configuration:**
   ```bash
   # Check JWT secret
   kubectl get secret jwt-secret -n api-gateway
   
   # Verify token validation
   curl -H "Authorization: Bearer <token>" http://api-gateway/api/validate
   ```

**Resolution Steps:**

1. **Update JWT secret if needed:**
   ```bash
   kubectl create secret generic jwt-secret \
     --from-literal=secret=<new-secret> \
     -n api-gateway
   ```

2. **Restart API Gateway:**
   ```bash
   kubectl rollout restart deployment/api-gateway -n api-gateway
   ```

## Monitoring and Alerting

### Key Metrics to Monitor:
- Request rate (requests/second)
- Error rate (4xx, 5xx responses)
- Response time (p50, p95, p99)
- Upstream service health
- Resource utilization (CPU, memory)

### Alert Thresholds:
- Error rate > 5%
- Response time p95 > 2 seconds
- Upstream service down
- CPU usage > 80%

### Dashboard Queries:
```promql
# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Upstream health
up{job="api-gateway"}
```

## Escalation Procedures

### Level 1 (On-call Engineer):
- Check basic service health
- Verify configuration
- Restart services if needed
- Escalate if issue persists > 15 minutes

### Level 2 (Senior Engineer):
- Deep dive into logs and metrics
- Analyze performance bottlenecks
- Coordinate with other teams
- Escalate if issue persists > 1 hour

### Level 3 (Principal Engineer):
- Architecture review
- Cross-team coordination
- Post-incident analysis
- Long-term solution design

## Post-Incident Actions

1. **Immediate (within 1 hour):**
   - Document incident timeline
   - Collect relevant logs and metrics
   - Notify stakeholders

2. **Short-term (within 24 hours):**
   - Conduct post-mortem meeting
   - Identify root cause
   - Implement immediate fixes

3. **Long-term (within 1 week):**
   - Update runbooks and procedures
   - Implement monitoring improvements
   - Plan architectural improvements

## Related Documentation

- [API Gateway Architecture](../architecture/api-gateway.md)
- [Load Balancer Configuration](../networking/load-balancer.md)
- [Database Connection Pooling](../database/connection-pooling.md)
- [Authentication Service](../auth/oauth-integration.md)

## Contact Information

- **Primary On-call:** platform-oncall@company.com
- **Secondary:** platform-team@company.com
- **Emergency:** +1-555-PLATFORM
- **Slack:** #platform-incidents
