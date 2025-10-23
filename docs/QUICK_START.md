# Quick Start Guide

Get the Ops Runbook RAG system up and running in 30 minutes for research and evaluation.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Git
- 8GB+ RAM recommended

## 1. Clone and Setup (5 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd ops-runbook-rag

# Copy environment configuration
cp .env.example .env

# Edit environment variables (optional)
# Add your API keys for OpenAI, Anthropic, Cohere
nano .env
```

## 2. Start Research Environment (10 minutes)

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f backend

# Verify all services are running
docker-compose ps
```

**Expected Services:**
- ✅ Backend API (port 8000)
- ✅ PostgreSQL with pgvector (port 5432)
- ✅ Redis cache (port 6379)
- ✅ Prometheus metrics (port 9090)
- ✅ Grafana dashboards (port 3000)
- ✅ Jaeger tracing (port 16686)

## 3. Initialize Database (5 minutes)

```bash
# Run database initialization
python scripts/setup_research_env.py

# Verify database setup
curl http://localhost:8000/api/v1/health
```

**Expected Output:**
```
✅ Database connection successful
✅ pgvector extension available
✅ Cache connection successful
✅ Database initialized successfully
🎉 Research environment setup completed successfully!
```

## 4. Ingest Sample Data (5 minutes)

```bash
# Ingest sample runbooks and KB articles
python scripts/ingest_batch.py --data-dir sample_data/ --monitor

# Check ingestion status
curl http://localhost:8000/api/v1/ingest/stats
```

**Expected Output:**
```
✅ Ingested 1 runbooks
✅ Ingested 1 KB articles
🎉 All tasks completed successfully!
```

## 5. Test the System (5 minutes)

### Test Query Endpoint

```bash
# Test a sample query
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API Gateway is returning 503 errors, what should I do?",
    "context": {
      "service": "api-gateway",
      "severity": "critical"
    }
  }'
```

### Test Health Endpoints

```bash
# System health
curl http://localhost:8000/api/v1/health

# System statistics
curl http://localhost:8000/api/v1/health/stats

# Query analytics
curl http://localhost:8000/api/v1/query/analytics
```

## 6. Explore the Interface

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/research)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### Sample Queries to Try

1. **Service Issues:**
   ```json
   {
     "query": "How to troubleshoot database performance issues?",
     "context": {"service": "database", "severity": "high"}
   }
   ```

2. **Authentication Problems:**
   ```json
   {
     "query": "API authentication is failing, what should I check?",
     "context": {"service": "auth-service", "component": "authentication"}
   }
   ```

3. **Deployment Issues:**
   ```json
   {
     "query": "How to rollback a failed deployment?",
     "context": {"service": "deployment", "incident_type": "deployment_issues"}
   }
   ```

## 7. Run Evaluation (Optional)

```bash
# Run evaluation on curated incident set
python scripts/evaluate.py \
  --dataset sample_data/evaluation/incident_set.json \
  --output evaluation_results.json

# View results
cat evaluation_results.json | jq '.metrics'
```

## Troubleshooting

### Common Issues

**1. Services not starting:**
```bash
# Check Docker logs
docker-compose logs backend
docker-compose logs postgres

# Restart services
docker-compose restart
```

**2. Database connection issues:**
```bash
# Check PostgreSQL status
docker-compose exec postgres psql -U research -d ops_runbook_rag -c "SELECT 1;"

# Verify pgvector extension
docker-compose exec postgres psql -U research -d ops_runbook_rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

**3. API not responding:**
```bash
# Check backend logs
docker-compose logs -f backend

# Test API directly
curl http://localhost:8000/health
```

**4. Memory issues:**
```bash
# Increase Docker memory limit
# Edit docker-compose.yml and increase memory limits
# Or use: docker-compose up --scale backend=1
```

### Reset Environment

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Start fresh
docker-compose up -d
python scripts/setup_research_env.py
```

## Next Steps

### For Researchers

1. **Explore the API**: Use the Swagger UI to understand all endpoints
2. **Run Evaluations**: Use the evaluation script with your own datasets
3. **Monitor Performance**: Check Grafana dashboards for system metrics
4. **Analyze Results**: Review evaluation metrics and system performance

### For Developers

1. **Add Custom Parsers**: Extend the ingestion pipeline for new document types
2. **Implement Retrieval Methods**: Add new retrieval strategies in the retrieval module
3. **Enhance Validation**: Improve response validation for your use cases
4. **Add Metrics**: Implement custom metrics for your research needs

### For Operations Teams

1. **Ingest Your Runbooks**: Add your own operational documentation
2. **Configure Monitoring**: Set up alerts based on system performance
3. **Customize Prompts**: Adapt incident-aware prompting for your environment
4. **Scale the System**: Deploy to production using Kubernetes manifests

## Support

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report problems in the issue tracker
- **Research**: Check `evaluation/` directory for research datasets
- **API Reference**: Use Swagger UI at http://localhost:8000/docs

## Performance Expectations

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 10GB for sample data and embeddings

### Performance Metrics
- **Query Latency**: < 3 seconds (P95)
- **Ingestion Speed**: ~10 documents/minute
- **Memory Usage**: ~2GB for full sample dataset
- **Storage**: ~1GB for embeddings and metadata

### Scaling Considerations
- **Horizontal**: Scale backend replicas
- **Vertical**: Increase memory for larger datasets
- **Caching**: Redis improves repeated query performance
- **Database**: PostgreSQL can handle 100K+ documents

---

**🎉 You're ready to start researching with the Ops Runbook RAG system!**

For detailed information, see:
- [Architecture Guide](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Runbook Authoring Guide](RUNBOOK_AUTHORING.md)
