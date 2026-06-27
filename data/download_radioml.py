"""Download RadioML 2016.10A dataset from multiple sources."""
import urllib.request
import ssl
import tarfile
import os

dest_dir = os.path.dirname(os.path.abspath(__file__))
archive = os.path.join(dest_dir, "RML2016.10a.tar.bz2")
target_pkl = os.path.join(dest_dir, "RML2016.10a_dict.pkl")

if os.path.exists(target_pkl):
    print(f"Already exists: {target_pkl}")
    exit(0)

# SSL context that skips verification (DeepSig's cert is expired)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

urls = [
    # HTTP (no SSL)
    "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
    # HTTPS with SSL bypass
    "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
]

for url in urls:
    print(f"Trying: {url}")
    try:
        if url.startswith("https"):
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, context=ctx)
        else:
            resp = urllib.request.urlopen(url)
        
        with open(archive, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"  Downloaded {total / 1e6:.1f} MB...", end="\r")
        resp.close()
        size = os.path.getsize(archive)
        print(f"\nDownloaded: {archive} ({size / 1e6:.1f} MB)")
        if size < 1000:
            print("  File too small, skipping...")
            os.remove(archive)
            continue
        break
    except Exception as e:
        print(f"  Failed: {e}")
        if os.path.exists(archive):
            os.remove(archive)
else:
    print("\nAll URLs failed. Please download manually:")
    print("  1. Go to https://www.kaggle.com/datasets/jchen2186/radioml-201610a")
    print("  2. Download RML2016.10a_dict.pkl")
    print(f"  3. Place it in: {dest_dir}")
    exit(1)

# Extract
print("Extracting...")
with tarfile.open(archive, "r:bz2") as tar:
    tar.extractall(path=dest_dir)
print(f"Extracted to {dest_dir}")

os.remove(archive)
print("Done!")
for f in os.listdir(dest_dir):
    print(f"  {f}")
