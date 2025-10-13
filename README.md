python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows powershell
# .venv\Scripts\Activate.ps1
------

pip install -r requirements.txt

------
python src/diaclinic/train.py

------
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
