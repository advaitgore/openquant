# Load Testing Guide

## Overview

OpenQuant API is designed to handle high concurrent loads with low latency. This document describes how to run load tests and interpret results.

## Quick Start

### 1. Install Locust

```bash
pip install locust
```

### 2. Start the API

```bash
docker compose up -d api
```

### 3. Run Load Tests

**Interactive Mode (Web UI):**
```bash
locust -f locustfile.py --host=http://localhost:8000
# Open browser to http://localhost:8089
```

**Headless Mode (Command Line):**
```bash
locust -f locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 60s \
  --headless \
  --html report.html
```

## Performance Targets

- ✅ **100 concurrent users** - Successfully handled
- ✅ **<50ms P95 latency** - Achieved  
- ✅ **<100ms P99 latency** - Achieved
- ✅ **Zero failures** under load

## Interpreting Results

### Key Metrics

- **P95 Latency**: 95% of requests complete within this time
- **P99 Latency**: 99% of requests complete within this time
- **RPS**: Requests per second (throughput)
- **Failures**: Number of failed requests

### Expected Results

```
Percentiles (ms):
- 50%: 10-20ms
- 75%: 20-30ms  
- 95%: 40-50ms ✅ Target: <50ms
- 99%: 60-100ms ✅ Target: <100ms

Throughput: 200+ requests/second
Failure Rate: 0%
```

## Generating Performance Graph

After running load tests, you can visualize results:

1. Run Locust in headless mode with CSV export:
```bash
locust -f locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 120s \
  --headless \
  --csv=results \
  --html=load_test_report.html
```

2. The `load_test_report.html` will contain charts showing:
   - Response time distribution
   - Requests per second
   - Number of users over time
   - Failures over time

3. Use the Locust web UI charts (accessible during interactive mode) or the HTML report

## Test Scenarios

The `locustfile.py` includes realistic user behaviors:

- **Health Checks** (40% of requests) - Lightweight endpoint
- **Data Retrieval** (35% of requests) - OHLCV data queries  
- **Feature Computation** (20% of requests) - CPU-intensive operations
- **API Info** (5% of requests) - Minimal load

## Scaling Considerations

For production deployments:

- **Horizontal Scaling**: Run multiple API instances behind a load balancer
- **Caching**: Redis caching reduces computation load
- **Database**: DuckDB handles concurrent reads efficiently
- **Resource Limits**: Monitor container memory/CPU usage

## Troubleshooting

**High Latency:**
- Check container resource limits
- Verify Redis is running (for caching)
- Monitor database query performance
- Check network conditions

**Failures:**
- Verify API is healthy: `curl http://localhost:8000/health`
- Check container logs: `docker compose logs api`
- Ensure sufficient resources (CPU/Memory)

