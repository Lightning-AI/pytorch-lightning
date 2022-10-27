# app.py
import lightning as L

class LitWorker(L.LightningWork):
   def run(self):
      # any executables and even other languages like R and Julia
      subprocess.run(['/bin/bash', 'start_script.sh'])
      subprocess.run(['R', 'scriptName.R'])

# run on 1 cloud CPU
compute = L.CloudCompute("cpu")
app = L.LightningApp(LitWorker(cloud_compute=compute))
