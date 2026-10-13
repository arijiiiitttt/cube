# NeonDB Setup Guide — Cube

## Step 1 — Create Account & Project
1. Go to **https://neon.tech** → Sign up free (GitHub login works)
2. Click **New Project** → name it `cube` → pick closest region → **Create**

## Step 2 — Get Connection String
1. Dashboard → **Connection Details** → copy the connection string
2. It looks like:
   ```
   postgresql://alex:AbCdEf@ep-cool-name-123456.us-east-2.aws.neon.tech/cube?sslmode=require
   ```
3. For FastAPI (asyncpg driver), change prefix:
   ```
   postgresql+asyncpg://alex:AbCdEf@ep-cool-name-123456.us-east-2.aws.neon.tech/cube?sslmode=require
   ```

## Step 3 — Run the SQL (ONE command)
1. Dashboard → **SQL Editor** (left sidebar)
2. Open `database/neondb_setup.sql` from this project
3. Copy the **entire single line** → paste → click **Run**
4. Verify output shows 4 tables with 0 rows each ✓

## Step 4 — Add to .env
```env
DATABASE_URL=postgresql+asyncpg://YOUR_USER:YOUR_PASS@YOUR_HOST/cube?sslmode=require
```

## Step 5 — Verify
Start the backend and visit http://localhost:8000/health
```json
{"status": "ok", "service": "Cube API", "version": "0.1.0"}
```

## Tables Created

| Table            | Purpose                                           |
|------------------|---------------------------------------------------|
| `datasets`       | Uploaded files, token counts, processing status   |
| `training_jobs`  | Job status, epoch/step/loss, model path           |
| `model_registry` | Trained models, metrics, API keys, endpoints      |

## NeonDB Free Tier Limits
- Storage: **512 MB** (enough for hundreds of training runs worth of metadata)
- Compute: 0.25 vCPU, auto-suspends after 5 min inactivity
- Branches: 10 (great for dev/staging/prod separation)
- **No credit card required**

## Troubleshooting

**SSL error**: Make sure your DATABASE_URL includes `?sslmode=require`

**asyncpg error**: Make sure you're using `postgresql+asyncpg://` not `postgresql://`

**Connection refused**: NeonDB auto-suspends — first connection after idle takes ~2s to wake up. This is normal.
