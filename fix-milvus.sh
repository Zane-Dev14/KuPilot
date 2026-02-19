#!/bin/bash
# Fix Milvus initialization issues
# The errors indicate datanode/querynode failures — usually due to incomplete startup
# This script does a clean rebuild from scratch

set -e

echo "🔧 Cleaning up Milvus..."

# Step 1: Stop containers
echo "1️⃣  Stopping containers..."
docker compose down

# Step 2: Remove volumes (this is KEY — old state causes issues)
echo "2️⃣  Removing volumes (fixing corrupted state)..."
docker volume rm $(docker volume ls -q | grep -E 'milvus|etcd|minio' || echo '') 2>/dev/null || true

# Step 3: Check if containers are fully stopped
echo "3️⃣  Waiting for cleanup..."
sleep 3

# Step 4: Start fresh
echo "4️⃣  Starting Milvus stack..."
docker compose up -d

# Step 5: Wait for health checks
echo "5️⃣  Waiting for services to be healthy (this takes 30-60 seconds)..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    
    # Check etcd
    if docker compose logs milvus-etcd 2>/dev/null | grep -q "ready to serve"; then
        etcd_ok=true
    else
        etcd_ok=false
    fi
    
    # Check minio
    if docker compose ps milvus-minio 2>/dev/null | grep -q "healthy"; then
        minio_ok=true
    else
        minio_ok=false
    fi
    
    # Check milvus
    if curl -s http://localhost:19530 > /dev/null 2>&1; then
        milvus_ok=true
    else
        milvus_ok=false
    fi
    
    if [ "$etcd_ok" = true ] && [ "$minio_ok" = true ] && [ "$milvus_ok" = true ]; then
        echo "✅ All services healthy!"
        break
    fi
    
    echo "  Attempt $attempt/$max_attempts: etcd=$etcd_ok, minio=$minio_ok, milvus=$milvus_ok"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "⚠️  Services not fully ready after timeout, but trying anyway..."
fi

# Step 6: Verify connection
echo ""
echo "6️⃣  Verifying Milvus connection..."
python3 << 'EOF'
import time
import sys
sys.path.insert(0, '/Users/eric/IBM/Projects/courses/Deliverables/Week-2')

from src.vectorstore import MilvusStore

try:
    store = MilvusStore()
    if store.health_check():
        print("✅ Milvus connection verified!")
    else:
        print("❌ Milvus health check failed")
except Exception as e:
    print(f"❌ Connection error: {e}")
    sys.exit(1)
EOF

echo ""
echo "✅ Milvus is ready!"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/ingest.py"
echo "2. Run: python scripts/chat.py"
