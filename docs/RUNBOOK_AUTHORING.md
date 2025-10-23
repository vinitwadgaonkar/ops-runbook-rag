# Runbook Authoring Guide

This guide provides best practices for writing operational runbooks that work effectively with the RAG system.

## Overview

Well-structured runbooks are essential for effective incident response and knowledge management. This guide covers:

- Runbook structure and formatting
- Metadata requirements
- Code block best practices
- Section organization
- Research-specific features

## Runbook Structure

### 1. Frontmatter (Required)

Every runbook must start with YAML frontmatter containing metadata:

```yaml
---
service: api-gateway
severity: critical
component: ingress
team: platform
tags: [troubleshooting, networking, load-balancing]
last_updated: 2024-01-15
version: 2.1
---
```

**Required Fields**:
- `service`: Service name (e.g., "api-gateway", "database")
- `severity`: Incident severity ("critical", "high", "medium", "low")
- `component`: System component ("ingress", "database", "application")

**Optional Fields**:
- `team`: Responsible team
- `tags`: Searchable tags
- `last_updated`: Last modification date
- `version`: Runbook version
- `environment`: Target environment
- `incident_type`: Type of incident

### 2. Document Structure

```markdown
# Title

## Overview
Brief description of the issue and when to use this runbook.

## Symptoms
What to look for - error messages, metrics, behaviors.

## Diagnosis Steps
Step-by-step troubleshooting process.

## Resolution Steps
How to fix the issue.

## Prevention
How to prevent this issue in the future.

## Related Documentation
Links to other relevant runbooks.
```

## Section Guidelines

### Overview Section

**Purpose**: Provide context and scope for the runbook.

**Best Practices**:
- Start with a clear problem statement
- Define when to use this runbook
- Include affected systems and users
- Mention expected resolution time

**Example**:
```markdown
## Overview

This runbook addresses 503 Service Unavailable errors from the API Gateway. 
Use this runbook when:
- Clients receive 503 errors
- Error rates exceed 5%
- Load balancer health checks fail

**Affected Systems**: API Gateway, Load Balancer, Upstream Services
**Expected Resolution**: 15-30 minutes
**Escalation**: If unresolved after 30 minutes, escalate to Senior Engineer
```

### Symptoms Section

**Purpose**: Help operators quickly identify the issue.

**Best Practices**:
- List specific error messages
- Include metric thresholds
- Mention monitoring dashboards
- Provide quick diagnostic commands

**Example**:
```markdown
## Symptoms

**Error Messages**:
- HTTP 503 Service Unavailable
- "upstream connect error or disconnect/reset before headers"
- "no healthy upstream"

**Metrics to Check**:
- Error rate > 5% in last 5 minutes
- Upstream service health < 100%
- Load balancer backend errors > 0

**Quick Check**:
```bash
# Check error rate
curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])

# Check upstream health
kubectl get endpoints api-gateway-service
```
```

### Diagnosis Steps

**Purpose**: Systematic troubleshooting process.

**Best Practices**:
- Start with quick checks
- Progress to detailed analysis
- Include commands and expected outputs
- Provide decision trees

**Example**:
```markdown
## Diagnosis Steps

### Step 1: Check Service Health
```bash
# Check if pods are running
kubectl get pods -n api-gateway

# Check pod status
kubectl describe pods -n api-gateway
```

**Expected Output**: All pods should be "Running"
**If Not**: Proceed to Step 2

### Step 2: Check Resource Usage
```bash
# Check CPU and memory
kubectl top pods -n api-gateway

# Check for resource limits
kubectl describe pod <pod-name> -n api-gateway
```

**Expected Output**: CPU < 80%, Memory < 80%
**If High**: Proceed to Step 3
```

### Resolution Steps

**Purpose**: Provide clear, actionable steps to fix the issue.

**Best Practices**:
- Number steps sequentially
- Include verification steps
- Provide rollback procedures
- Include safety considerations

**Example**:
```markdown
## Resolution Steps

### Option 1: Restart Unhealthy Pods
```bash
# Delete unhealthy pods (they will be recreated)
kubectl delete pod <unhealthy-pod-name> -n api-gateway

# Verify new pods are healthy
kubectl get pods -n api-gateway
```

### Option 2: Scale Up Service
```bash
# Increase replica count
kubectl scale deployment api-gateway --replicas=3 -n api-gateway

# Verify scaling
kubectl get deployment api-gateway -n api-gateway
```

### Verification
```bash
# Check error rate is decreasing
curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[1m])

# Test endpoint directly
curl -I http://api-gateway/api/health
```
```

## Code Block Best Practices

### 1. Language Specification

Always specify the language for code blocks:

```bash
# Shell commands
kubectl get pods -n api-gateway
```

```yaml
# Configuration files
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
```

```sql
-- Database queries
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

### 2. Command Sequences

Group related commands together:

```bash
# Check service status
kubectl get pods -n api-gateway
kubectl get services -n api-gateway
kubectl get ingress -n api-gateway

# Check logs
kubectl logs -f deployment/api-gateway -n api-gateway
```

### 3. Expected Outputs

Include expected outputs for verification:

```bash
# Check pod status
kubectl get pods -n api-gateway

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# api-gateway-7d4b8c9f6-abc123   1/1     Running   0          5m
# api-gateway-7d4b8c9f6-def456   1/1     Running   0          5m
```

### 4. Error Handling

Include error scenarios and responses:

```bash
# Check service health
curl http://api-gateway/api/health

# Success response (200):
# {"status": "healthy", "timestamp": "2024-01-15T10:30:00Z"}

# Error response (503):
# {"error": "Service Unavailable", "code": 503}
```

## Metadata Best Practices

### 1. Service Naming

Use consistent service names:
- `api-gateway` (not "API Gateway" or "api_gateway")
- `user-service` (not "UserService" or "user service")
- `database` (not "PostgreSQL" or "DB")

### 2. Severity Levels

Use standard severity levels:
- `critical`: Service completely down
- `high`: Significant impact, workaround available
- `medium`: Limited impact, normal operations affected
- `low`: Minor impact, monitoring required

### 3. Component Classification

Use consistent component names:
- `ingress`: Load balancer, API Gateway
- `database`: PostgreSQL, MySQL, Redis
- `application`: Business logic services
- `infrastructure`: Kubernetes, networking

### 4. Tagging Strategy

Use descriptive tags for searchability:
```yaml
tags: [troubleshooting, networking, load-balancing, kubernetes]
tags: [database, performance, postgresql, optimization]
tags: [authentication, jwt, oauth, security]
```

## Research-Specific Features

### 1. Incident Type Classification

Classify runbooks by incident type for better retrieval:

```yaml
incident_type: service_unavailable
incident_type: performance_degradation
incident_type: authentication_failure
incident_type: deployment_issue
incident_type: resource_exhaustion
```

### 2. Complexity Scoring

The system automatically calculates complexity scores based on:
- Number of steps
- Code block count
- Section depth
- Content length

### 3. Actionability Indicators

Include actionability keywords for better retrieval:
- "check", "verify", "run", "execute"
- "restart", "scale", "deploy"
- "kubectl", "docker", "curl"

### 4. Temporal Context

Include time-sensitive information:
```markdown
## Time-Sensitive Actions

**Immediate (0-5 minutes)**:
- Check service health
- Verify basic connectivity

**Short-term (5-15 minutes)**:
- Analyze logs and metrics
- Implement quick fixes

**Long-term (15+ minutes)**:
- Root cause analysis
- Permanent solution implementation
```

## Quality Checklist

### Before Publishing

- [ ] Frontmatter includes all required fields
- [ ] Symptoms section helps with quick identification
- [ ] Diagnosis steps are logical and sequential
- [ ] Resolution steps are clear and actionable
- [ ] Code blocks have proper language specification
- [ ] Commands include expected outputs
- [ ] Error scenarios are covered
- [ ] Verification steps are included
- [ ] Related documentation is linked
- [ ] Tags are relevant and searchable

### Testing

- [ ] All commands have been tested
- [ ] Expected outputs are accurate
- [ ] Error scenarios are realistic
- [ ] Resolution steps actually work
- [ ] Verification steps are reliable

## Examples

### Good Runbook Example

```markdown
---
service: api-gateway
severity: critical
component: ingress
team: platform
tags: [troubleshooting, networking, load-balancing]
---

# API Gateway 503 Errors

## Overview
This runbook addresses 503 Service Unavailable errors from the API Gateway.

## Symptoms
- HTTP 503 errors in client applications
- Error rate > 5% in monitoring
- Load balancer health checks failing

## Diagnosis Steps

### 1. Check Service Health
```bash
kubectl get pods -n api-gateway
kubectl get services -n api-gateway
```

### 2. Check Resource Usage
```bash
kubectl top pods -n api-gateway
kubectl describe pod <pod-name> -n api-gateway
```

## Resolution Steps

### Option 1: Restart Unhealthy Pods
```bash
kubectl delete pod <unhealthy-pod-name> -n api-gateway
```

### Option 2: Scale Up Service
```bash
kubectl scale deployment api-gateway --replicas=3 -n api-gateway
```

## Verification
```bash
curl -I http://api-gateway/api/health
```

## Related Documentation
- [Load Balancer Configuration](../networking/load-balancer.md)
- [Kubernetes Troubleshooting](../kubernetes/troubleshooting.md)
```

### Poor Runbook Example

```markdown
# API Gateway Issues

Sometimes the API Gateway doesn't work. Here's what to do:

1. Check if it's working
2. If not, try to fix it
3. If that doesn't work, ask someone else

Good luck!
```

## Advanced Features

### 1. Conditional Steps

Use conditional logic for different scenarios:

```markdown
## Resolution Steps

### If Error Rate < 50%
```bash
# Restart individual pods
kubectl delete pod <pod-name> -n api-gateway
```

### If Error Rate > 50%
```bash
# Scale up entire service
kubectl scale deployment api-gateway --replicas=5 -n api-gateway
```
```

### 2. Escalation Procedures

Include clear escalation paths:

```markdown
## Escalation

**Level 1 (0-15 minutes)**:
- Check basic service health
- Restart unhealthy pods
- Verify configuration

**Level 2 (15-30 minutes)**:
- Deep dive into logs
- Check upstream dependencies
- Coordinate with other teams

**Level 3 (30+ minutes)**:
- Architecture review
- Cross-team coordination
- Post-incident analysis
```

### 3. Monitoring Integration

Include monitoring queries and dashboards:

```markdown
## Monitoring

**Key Metrics**:
- Error rate: `rate(http_requests_total{status=~"5.."}[5m])`
- Response time: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- Upstream health: `up{job="api-gateway"}`

**Dashboards**:
- API Gateway Overview: http://grafana/d/api-gateway
- Load Balancer Health: http://grafana/d/load-balancer
```

## Conclusion

Well-structured runbooks are essential for effective incident response and knowledge management. By following these guidelines, you can create runbooks that:

- Are easily discoverable by the RAG system
- Provide clear, actionable guidance
- Support both human operators and AI systems
- Enable continuous improvement through feedback

Remember: Good runbooks save time, reduce errors, and improve system reliability. Invest in their quality for long-term operational success.

---

**For more information, see:**
- [Quick Start Guide](QUICK_START.md)
- [Architecture Guide](ARCHITECTURE.md)
- [API Documentation](API.md)
