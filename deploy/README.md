# Deployment Guide

## VPS Setup

### 1. Clone the repository

```bash
sudo mkdir -p /opt/adnihilator
sudo chown $USER:$USER /opt/adnihilator
git clone https://github.com/your-repo/adnihilator.git /opt/adnihilator
```

### 2. Create virtual environment

```bash
cd /opt/adnihilator
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 3. Create data directory

```bash
sudo mkdir -p /var/lib/adnihilator
sudo chown www-data:www-data /var/lib/adnihilator
```

### 4. Configure environment

```bash
sudo mkdir -p /etc/adnihilator
sudo nano /etc/adnihilator/env
```

Add:
```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<generate-strong-password>
WORKER_API_KEY=<generate-random-key>
DATABASE_PATH=/var/lib/adnihilator/adnihilator.db
R2_PUBLIC_URL=https://your-bucket.r2.dev
```

Secure the file:
```bash
sudo chmod 600 /etc/adnihilator/env
sudo chown root:www-data /etc/adnihilator/env
```

### 5. Install systemd service

```bash
sudo cp deploy/adnihilator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable adnihilator
sudo systemctl start adnihilator
```

### 6. Configure nginx

```bash
sudo cp deploy/nginx-site.conf /etc/nginx/sites-available/adnihilator
sudo ln -s /etc/nginx/sites-available/adnihilator /etc/nginx/sites-enabled/
# Edit the file to replace yourdomain.com with your actual domain
sudo nano /etc/nginx/sites-available/adnihilator
```

### 7. Get SSL certificate

```bash
sudo certbot --nginx -d feeds.yourdomain.com
```

### 8. Restart nginx

```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Local Worker Setup

### 1. Set environment variables

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export API_URL=https://feeds.yourdomain.com
export WORKER_API_KEY=<same-key-as-server>
export R2_ACCESS_KEY=<cloudflare-r2-key>
export R2_SECRET_KEY=<cloudflare-r2-secret>
export R2_BUCKET=adnihilator-audio
export R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
export OPENAI_API_KEY=<your-openai-key>
```

### 2. Run the worker

```bash
# One-shot mode (process one job)
adnihilator worker --once

# Daemon mode (continuous processing)
adnihilator worker --daemon --interval 300
```

### 3. (Optional) Configure launchd for auto-start

Create `~/Library/LaunchAgents/com.adnihilator.worker.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.adnihilator.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/adnihilator</string>
        <string>worker</string>
        <string>--daemon</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>API_URL</key>
        <string>https://feeds.yourdomain.com</string>
        <!-- Add other env vars -->
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
```
